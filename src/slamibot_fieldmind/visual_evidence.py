"""Bounded image evidence for a calling multimodal agent; no VLM inference here."""

MAX_EVIDENCE_IMAGES = 3


def attach_visual_evidence(
    result, rows, load_image, *, include_images=True, max_edge=1280
):
    if not 64 <= max_edge <= 1280:
        raise ValueError(
            "automatic evidence max_edge must be 64..1280; use get_frame_image for larger images"
        )
    images = []
    for index, row in enumerate(rows):
        evidence = {"status": "not_requested", "overlay": False}
        row["image_evidence"] = evidence
        row["visual_verification"] = "not_performed"
        if not include_images:
            continue
        if index >= MAX_EVIDENCE_IMAGES:
            evidence["status"] = "omitted_limit"
            continue
        try:
            jpeg = load_image(row, max_edge)
        except (OSError, ValueError) as error:
            evidence.update(status="unavailable", reason=str(error))
            continue
        evidence.update(status="attached", image_index=len(images) + 1)
        context = {
            "image_index": evidence["image_index"],
            "frame_id": row["frame_id"],
            "camera_id": row["camera_id"],
            "captured_at": row["captured_at"],
            "session_id": row.get("session_id"),
            "coordinate_id": row.get("coordinate_id"),
            "source": result.get("source", "historical_keyframe"),
            "stale": row.get("stale"),
            "overlay": False,
        }
        images.append({"context": context, "jpeg": jpeg})
    result["visual_review"] = {
        "status": "required"
        if images
        else "unavailable"
        if include_images and rows
        else "not_requested"
        if rows
        else "not_applicable",
        "performed": False,
        "reviewer": "calling_multimodal_agent",
        "attached_images": len(images),
        "image_limit": MAX_EVIDENCE_IMAGES,
        "instruction": (
            "Inspect the actual images before answering. RADIO ranks candidates; it does not confirm the requested object. "
            "YOLO labels/counts may be wrong or incomplete. Check category, attributes and counts visually; reject mismatches. "
            "If candidates do not show the target, report no confirmed evidence in the reviewed candidates, not absence everywhere. "
            "For scene descriptions, describe the full image and distinguish visual observations from YOLO detections. "
            "Do not claim visual confirmation if images are missing, unreadable or not understood. Respect source time and stale."
        ),
    }
    return result, images
