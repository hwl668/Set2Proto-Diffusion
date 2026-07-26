"""CelebA integrity audit and deterministic identity-disjoint split creation."""

from __future__ import annotations

import csv
import hashlib
import json
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping


PARTITION_NAMES = {0: "train", 1: "val", 2: "test"}
OFFICIAL_MD5 = {
    "Anno/identity_CelebA.txt": "32bd1bd63d3c78cd57e08160ec5ed1e2",
    "Anno/list_attr_celeba.txt": "75e246fa4810816ffd6ee81facbd244c",
    "Anno/list_bbox_celeba.txt": "00566efa6fedff7a56946cd1c10f1c16",
    "Anno/list_landmarks_align_celeba.txt": (
        "cc24ecafdb5b50baae59b03474781f8c"
    ),
    "Eval/list_eval_partition.txt": "d32c9cbf5e040fd4025c592c306e6668",
}
ALIGNED_ZIP_MD5 = "00d2c5bc6d35e252742224ab0c1e8fcb"
EXPECTED_RAW_VOLUME_COUNT = 14


@dataclass(frozen=True)
class RealDataPreparation:
    root: Path
    split_root: Path
    manifest_path: Path
    manifest: dict[str, Any]
    reused: bool


def _md5(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.md5(usedforsecurity=False)
    with path.open("rb") as handle:
        while chunk := handle.read(chunk_size):
            digest.update(chunk)
    return digest.hexdigest()


def _parse_two_column_file(path: Path) -> dict[str, int]:
    rows: dict[str, int] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            fields = line.split()
            if not fields:
                continue
            if len(fields) != 2:
                raise ValueError(
                    f"{path} line {line_number} must contain two fields"
                )
            image_name, value = fields
            if image_name in rows:
                raise ValueError(f"duplicate image name in {path}: {image_name}")
            rows[image_name] = int(value)
    return rows


def build_identity_split(
    identities_by_image: Mapping[str, int],
    partition_by_image: Mapping[str, int],
    *,
    requested_identities: Mapping[str, int],
    min_images_per_identity: int,
    seed: int,
    excluded_identities: Mapping[str, Iterable[int]] | None = None,
) -> tuple[dict[str, list[int]], dict[int, list[str]]]:
    """Select reproducible identity subsets from disjoint official partitions."""

    if set(identities_by_image) != set(partition_by_image):
        missing_partition = set(identities_by_image) - set(partition_by_image)
        missing_identity = set(partition_by_image) - set(identities_by_image)
        raise ValueError(
            "identity/partition image names differ: "
            f"missing_partition={len(missing_partition)}, "
            f"missing_identity={len(missing_identity)}"
        )

    images_by_identity: dict[int, list[str]] = defaultdict(list)
    partitions_by_identity: dict[int, set[int]] = defaultdict(set)
    for image_name, identity_id in identities_by_image.items():
        partition = int(partition_by_image[image_name])
        if partition not in PARTITION_NAMES:
            raise ValueError(f"invalid CelebA partition value: {partition}")
        identity_id = int(identity_id)
        images_by_identity[identity_id].append(image_name)
        partitions_by_identity[identity_id].add(partition)

    crossing = [
        identity_id
        for identity_id, partitions in partitions_by_identity.items()
        if len(partitions) != 1
    ]
    if crossing:
        raise ValueError(
            f"{len(crossing)} identities cross official partitions"
        )

    excluded = {
        split: {
            int(identity_id)
            for identity_id in (
                excluded_identities.get(split, ())
                if excluded_identities is not None
                else ()
            )
        }
        for split in ("train", "val", "test")
    }
    selected: dict[str, list[int]] = {}
    for partition, split in PARTITION_NAMES.items():
        eligible = sorted(
            identity_id
            for identity_id, images in images_by_identity.items()
            if len(images) >= min_images_per_identity
            and partitions_by_identity[identity_id] == {partition}
            and identity_id not in excluded[split]
        )
        random.Random(seed + partition).shuffle(eligible)
        requested = int(requested_identities[split])
        if len(eligible) < requested:
            raise ValueError(
                f"split {split} has {len(eligible)} eligible identities, "
                f"but {requested} were requested"
            )
        selected[split] = sorted(eligible[:requested])

    selected_sets = {split: set(values) for split, values in selected.items()}
    if (
        selected_sets["train"] & selected_sets["val"]
        or selected_sets["train"] & selected_sets["test"]
        or selected_sets["val"] & selected_sets["test"]
    ):
        raise RuntimeError("selected identity splits are not disjoint")

    for image_names in images_by_identity.values():
        image_names.sort()
    return selected, dict(images_by_identity)


def load_split_identities(
    split_root: Path,
) -> dict[str, set[int]]:
    """Load identity IDs from a previously materialized split."""

    split_root = split_root.expanduser().resolve()
    path = split_root / "identities.csv"
    if not path.is_file():
        raise FileNotFoundError(f"split identities file not found: {path}")
    result = {split: set() for split in ("train", "val", "test")}
    with path.open("r", encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        required = {"split", "identity_id"}
        if reader.fieldnames is None or not required.issubset(
            reader.fieldnames
        ):
            raise ValueError(
                f"{path} must contain columns {sorted(required)}"
            )
        for row in reader:
            split = str(row["split"])
            if split not in result:
                raise ValueError(f"invalid split in {path}: {split}")
            identity_id = int(row["identity_id"])
            if identity_id in result[split]:
                raise ValueError(
                    f"duplicate identity {identity_id} in split {split}"
                )
            result[split].add(identity_id)
    return result


def _detect_image_assets(root: Path, expected_images: int) -> dict[str, Any]:
    aligned_directories = (
        root / "Img" / "img_align_celeba",
        root / "img_align_celeba",
    )
    aligned_zips = (
        root / "Img" / "img_align_celeba.zip",
        root / "img_align_celeba.zip",
    )
    aligned_directory = next(
        (path for path in aligned_directories if path.is_dir()),
        None,
    )
    aligned_zip = next((path for path in aligned_zips if path.is_file()), None)
    aligned_image_count = 0
    if aligned_directory is not None:
        aligned_image_count = sum(
            1
            for path in aligned_directory.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

    raw_volume_directory = root / "Img" / "img_celeba.7z"
    raw_volumes = (
        sorted(raw_volume_directory.glob("img_celeba.7z.*"))
        if raw_volume_directory.is_dir()
        else []
    )
    aligned_zip_md5 = _md5(aligned_zip) if aligned_zip is not None else None
    aligned_zip_valid = aligned_zip_md5 == ALIGNED_ZIP_MD5
    aligned_directory_valid = aligned_image_count == expected_images
    return {
        "aligned_directory": (
            str(aligned_directory) if aligned_directory is not None else None
        ),
        "aligned_image_count": aligned_image_count,
        "aligned_directory_complete": aligned_directory_valid,
        "aligned_zip": str(aligned_zip) if aligned_zip is not None else None,
        "aligned_zip_md5": aligned_zip_md5,
        "aligned_zip_complete": aligned_zip_valid,
        "raw_volume_directory": (
            str(raw_volume_directory)
            if raw_volume_directory.is_dir()
            else None
        ),
        "raw_volume_count": len(raw_volumes),
        "raw_volume_expected": EXPECTED_RAW_VOLUME_COUNT,
        "raw_volumes_complete": len(raw_volumes) == EXPECTED_RAW_VOLUME_COUNT,
        "usable_aligned_images": aligned_directory_valid or aligned_zip_valid,
    }


def _write_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, Any]],
) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _identity_partitions(
    identities_by_image: Mapping[str, int],
    partition_by_image: Mapping[str, int],
) -> dict[int, set[int]]:
    result: dict[int, set[int]] = defaultdict(set)
    for image_name, identity_id in identities_by_image.items():
        result[int(identity_id)].add(int(partition_by_image[image_name]))
    return dict(result)


def prepare_celeba_metadata(
    *,
    root: Path,
    config: Mapping[str, Any],
    profile: str,
    resume: bool = False,
    exclude_split_root: Path | None = None,
) -> RealDataPreparation:
    """Audit official annotations and write deterministic split manifests."""

    root = root.expanduser().resolve()
    if not root.is_dir():
        raise FileNotFoundError(f"CelebA root does not exist: {root}")
    real_config = config["data"]["real"]
    expected_images = int(real_config["expected_images"])
    expected_identities = int(real_config["expected_identities"])
    seed = int(config["project"]["seed"])
    split_root = root / "splits" / f"{profile}-seed{seed}"
    if split_root.exists() and any(split_root.iterdir()) and not resume:
        raise FileExistsError(
            f"real split already exists; pass --resume to refresh: {split_root}"
        )
    existing_manifest_path = split_root / "manifest.json"
    if resume and existing_manifest_path.is_file():
        existing_manifest = json.loads(
            existing_manifest_path.read_text(encoding="utf-8")
        )
        existing_exclusion = (
            existing_manifest.get("selection", {})
            .get("exclusion", {})
            .get("split_root")
        )
        if exclude_split_root is None and existing_exclusion is not None:
            exclude_split_root = Path(existing_exclusion)
        elif (
            exclude_split_root is not None
            and existing_exclusion is not None
            and exclude_split_root.expanduser().resolve()
            != Path(existing_exclusion).expanduser().resolve()
        ):
            raise ValueError(
                "resume exclusion split differs from existing manifest"
            )
    split_root.mkdir(parents=True, exist_ok=True)

    hash_results: dict[str, dict[str, Any]] = {}
    for relative, expected_hash in OFFICIAL_MD5.items():
        path = root / Path(relative)
        actual_hash = _md5(path) if path.is_file() else None
        hash_results[relative] = {
            "exists": path.is_file(),
            "expected_md5": expected_hash,
            "actual_md5": actual_hash,
            "matches": actual_hash == expected_hash,
        }

    identities_by_image = _parse_two_column_file(
        root / "Anno" / "identity_CelebA.txt"
    )
    partition_by_image = _parse_two_column_file(
        root / "Eval" / "list_eval_partition.txt"
    )
    requested = {
        "train": int(real_config["train_identities"]),
        "val": int(real_config["val_identities"]),
        "test": int(real_config["test_identities"]),
    }
    excluded_identities = (
        load_split_identities(exclude_split_root)
        if exclude_split_root is not None
        else {split: set() for split in ("train", "val", "test")}
    )
    selected, images_by_identity = build_identity_split(
        identities_by_image,
        partition_by_image,
        requested_identities=requested,
        min_images_per_identity=int(real_config["min_images_per_identity"]),
        seed=seed,
        excluded_identities=excluded_identities,
    )

    identity_rows: list[dict[str, Any]] = []
    image_rows: list[dict[str, Any]] = []
    selected_summary: dict[str, dict[str, int]] = {}
    for split in ("train", "val", "test"):
        for identity_id in selected[split]:
            image_names = images_by_identity[identity_id]
            identity_rows.append(
                {
                    "split": split,
                    "identity_id": identity_id,
                    "image_count": len(image_names),
                }
            )
            for image_name in image_names:
                image_rows.append(
                    {
                        "split": split,
                        "identity_id": identity_id,
                        "image_name": image_name,
                        "official_partition": partition_by_image[image_name],
                    }
                )
        selected_summary[split] = {
            "identities": len(selected[split]),
            "images": sum(
                len(images_by_identity[identity_id])
                for identity_id in selected[split]
            ),
        }

    _write_csv(
        split_root / "identities.csv",
        ("split", "identity_id", "image_count"),
        identity_rows,
    )
    _write_csv(
        split_root / "images.csv",
        ("split", "identity_id", "image_name", "official_partition"),
        image_rows,
    )

    identity_counts = Counter(identities_by_image.values())
    partition_identities: dict[int, set[int]] = defaultdict(set)
    for image_name, identity_id in identities_by_image.items():
        partition_identities[partition_by_image[image_name]].add(identity_id)
    assets = _detect_image_assets(root, expected_images)
    official_partition_crossing = sum(
        len(partitions) > 1
        for partitions in _identity_partitions(
            identities_by_image,
            partition_by_image,
        ).values()
    )
    checks = {
        "official_annotation_hashes_match": all(
            value["matches"] for value in hash_results.values()
        ),
        "expected_image_annotations": len(identities_by_image) == expected_images,
        "expected_identities": len(identity_counts) == expected_identities,
        "identity_partition_names_match": (
            set(identities_by_image) == set(partition_by_image)
        ),
        "official_partitions_identity_disjoint": (
            official_partition_crossing == 0
        ),
        "selected_splits_identity_disjoint": (
            not (
                set(selected["train"]) & set(selected["val"])
                or set(selected["train"]) & set(selected["test"])
                or set(selected["val"]) & set(selected["test"])
            )
        ),
        "selected_identities_exclude_reference_split": all(
            not (set(selected[split]) & excluded_identities[split])
            for split in ("train", "val", "test")
        ),
        "usable_aligned_images_available": bool(
            assets["usable_aligned_images"]
        ),
    }
    manifest = {
        "schema_version": 1,
        "dataset": "celeba",
        "profile": profile,
        "seed": seed,
        "status": "passed" if all(checks.values()) else "blocked",
        "checks": checks,
        "annotation_hashes": hash_results,
        "assets": assets,
        "source": {
            "images": len(identities_by_image),
            "identities": len(identity_counts),
            "minimum_images": min(identity_counts.values()),
            "maximum_images": max(identity_counts.values()),
            "partition_identities": {
                PARTITION_NAMES[key]: len(value)
                for key, value in sorted(partition_identities.items())
            },
        },
        "selection": {
            "minimum_images_per_identity": int(
                real_config["min_images_per_identity"]
            ),
            "splits": selected_summary,
            "exclusion": {
                "split_root": (
                    str(exclude_split_root.expanduser().resolve())
                    if exclude_split_root is not None
                    else None
                ),
                "identities": {
                    split: len(excluded_identities[split])
                    for split in ("train", "val", "test")
                },
            },
        },
        "files": {
            "identities_csv": str(split_root / "identities.csv"),
            "images_csv": str(split_root / "images.csv"),
        },
    }
    manifest_path = split_root / "manifest.json"
    temporary_manifest = manifest_path.with_suffix(".json.tmp")
    temporary_manifest.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True),
        encoding="utf-8",
    )
    temporary_manifest.replace(manifest_path)
    return RealDataPreparation(
        root=root,
        split_root=split_root,
        manifest_path=manifest_path,
        manifest=manifest,
        reused=resume,
    )
