# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: OpenMDW-1.1

"""Ordered multi-view camera layout used by UR5 EEF training.

The layout is the single source of truth for:
  * which LeRobot video key is decoded for each canvas region
  * how tensors are vertically stacked / DROID-retiled
  * the human-readable view description appended to prompts
  * checkpoint / run metadata

Canonical close-desktop DROID layout (physical order):

  top          → observation.images.d435_rgb   → wrist
  bottom_left  → observation.images.d405_0_rgb → main
  bottom_right → observation.images.d405_1_rgb → side
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Literal, Mapping, Sequence

ViewLayout = Literal["vertical", "droid"]

# Region order for vertical stacking (top → bottom) and for DROID retiling
# (full-width top, then bottom-left, then bottom-right).
DROID_REGION_ORDER: tuple[str, ...] = ("top", "bottom_left", "bottom_right")
VERTICAL_REGION_ORDER: tuple[str, ...] = ("top", "middle", "bottom")


@dataclass(frozen=True)
class CameraView:
    """One physical camera bound to a canvas region."""

    region: str
    key: str
    label: str


@dataclass(frozen=True)
class CameraLayout:
    """Validated ordered camera layout for a three-view canvas."""

    views: tuple[CameraView, ...]
    view_layout: ViewLayout = "droid"

    def __post_init__(self) -> None:
        if len(self.views) != 3:
            raise ValueError(f"CameraLayout requires exactly 3 views, got {len(self.views)}")
        regions = [view.region for view in self.views]
        keys = [view.key for view in self.views]
        labels = [view.label for view in self.views]
        if len(set(regions)) != 3:
            raise ValueError(f"Duplicate regions in CameraLayout: {regions}")
        if len(set(keys)) != 3:
            raise ValueError(f"Duplicate camera keys in CameraLayout: {keys}")
        if len(set(labels)) != 3:
            raise ValueError(f"Duplicate labels in CameraLayout: {labels}")
        expected = DROID_REGION_ORDER if self.view_layout == "droid" else VERTICAL_REGION_ORDER
        if tuple(regions) != expected:
            raise ValueError(
                f"CameraLayout regions for view_layout={self.view_layout!r} must be "
                f"{list(expected)} in order, got {regions}"
            )
        for view in self.views:
            if not view.key or not view.label:
                raise ValueError(f"CameraView key/label must be non-empty: {view}")

    @property
    def ordered_keys(self) -> tuple[str, ...]:
        return tuple(view.key for view in self.views)

    @property
    def ordered_labels(self) -> tuple[str, ...]:
        return tuple(view.label for view in self.views)

    def label_for_region(self, region: str) -> str:
        for view in self.views:
            if view.region == region:
                return view.label
        raise KeyError(region)

    def key_for_region(self, region: str) -> str:
        for view in self.views:
            if view.region == region:
                return view.key
        raise KeyError(region)

    def description(self) -> str:
        """Human-readable layout text that matches physical tensor regions."""
        if self.view_layout == "droid":
            top, left, right = self.ordered_labels
            return (
                f"The top row is the {top} view. The bottom row contains two "
                f"horizontally concatenated views: the {left} view on the left and "
                f"the {right} view on the right."
            )
        top, middle, bottom = self.ordered_labels
        return (
            "The views are stacked vertically from top to bottom: "
            f"{top} view, {middle} view, {bottom} view."
        )

    def to_metadata(self) -> dict[str, Any]:
        return {
            "view_layout": self.view_layout,
            "views": [
                {"region": view.region, "key": view.key, "label": view.label}
                for view in self.views
            ],
            "description": self.description(),
            "physical_order": list(self.ordered_keys),
            "label_order": list(self.ordered_labels),
        }

    @classmethod
    def from_mapping(
        cls,
        views: Sequence[Mapping[str, str]] | Iterable[Mapping[str, str]],
        *,
        view_layout: ViewLayout = "droid",
    ) -> "CameraLayout":
        parsed = tuple(
            CameraView(
                region=str(item["region"]).strip(),
                key=str(item["key"]).strip(),
                label=str(item["label"]).strip(),
            )
            for item in views
        )
        return cls(views=parsed, view_layout=view_layout)


# Default close-desktop physical mapping used by the standalone template.
CLOSE_DESKTOP_DROID_LAYOUT = CameraLayout(
    views=(
        CameraView("top", "observation.images.d435_rgb", "wrist"),
        CameraView("bottom_left", "observation.images.d405_0_rgb", "main"),
        CameraView("bottom_right", "observation.images.d405_1_rgb", "side"),
    ),
    view_layout="droid",
)

CLOSE_DESKTOP_VERTICAL_LAYOUT = CameraLayout(
    views=(
        CameraView("top", "observation.images.d435_rgb", "wrist"),
        CameraView("middle", "observation.images.d405_0_rgb", "main"),
        CameraView("bottom", "observation.images.d405_1_rgb", "side"),
    ),
    view_layout="vertical",
)


def resolve_camera_layout(
    *,
    view_layout: ViewLayout = "droid",
    camera_layout: Sequence[Mapping[str, str]] | None = None,
) -> CameraLayout:
    """Build a CameraLayout from config, defaulting to close-desktop mapping."""
    if camera_layout is None:
        return (
            CLOSE_DESKTOP_DROID_LAYOUT
            if view_layout == "droid"
            else CLOSE_DESKTOP_VERTICAL_LAYOUT
        )
    return CameraLayout.from_mapping(camera_layout, view_layout=view_layout)
