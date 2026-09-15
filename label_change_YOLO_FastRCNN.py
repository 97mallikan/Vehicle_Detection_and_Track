#!/usr/bin/env python3

"""
convert_yolo_to_voc.py

Convert YOLO-format annotations into Pascal VOC XML annotations
for Faster R-CNN training.

YOLO input format:
    class_id x_center y_center width height

All YOLO coordinates are normalized between 0 and 1.

Output:
    One XML file for every image.

Example folder structure:

dataset/
├── images/
│   ├── frame001.jpg
│   ├── frame002.jpg
│   └── ...
│
├── labels/
│   ├── frame001.txt
│   ├── frame002.txt
│   └── ...
│
└── annotations_voc/
    ├── frame001.xml
    ├── frame002.xml
    └── ...

Run:
    python convert_yolo_to_voc.py
"""

from pathlib import Path
import xml.etree.ElementTree as ET

import cv2


# ============================================================
# CONFIGURATION
# ============================================================

IMAGE_DIR = Path("/media/anurag/nas-anurag/Dataset/Detection_Tracking.v1-detection_tracking_test.yolov8/train/images")
YOLO_LABEL_DIR = Path("/media/anurag/nas-anurag/Dataset/Detection_Tracking.v1-detection_tracking_test.yolov8/train/labels")

OUTPUT_DIR = Path("/media/anurag/nas-anurag/Dataset/Detection_Tracking.v1-detection_tracking_test.yolov8/train/annotations_voc")


# ------------------------------------------------------------
# IMPORTANT:
# The order MUST be identical to your YOLO training classes.
#
# Example from your vehicle dataset:
# ------------------------------------------------------------

CLASS_NAMES = [
    "car",
    "truck",
    "bus",
    "workvan",
    "motorcycle",
]


# Supported image types
IMAGE_EXTENSIONS = {
    ".jpg",
    ".jpeg",
    ".png",
    ".bmp",
    ".tif",
    ".tiff",
}


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def yolo_to_voc(
    x_center,
    y_center,
    box_width,
    box_height,
    image_width,
    image_height,
):
    """
    Convert normalized YOLO coordinates:

        xc, yc, width, height

    into Pascal VOC coordinates:

        xmin, ymin, xmax, ymax
    """

    x_center *= image_width
    y_center *= image_height

    box_width *= image_width
    box_height *= image_height

    xmin = x_center - box_width / 2.0
    ymin = y_center - box_height / 2.0

    xmax = x_center + box_width / 2.0
    ymax = y_center + box_height / 2.0

    # Clip to image boundaries
    xmin = max(0, min(image_width - 1, xmin))
    ymin = max(0, min(image_height - 1, ymin))

    xmax = max(0, min(image_width - 1, xmax))
    ymax = max(0, min(image_height - 1, ymax))

    return (
        int(round(xmin)),
        int(round(ymin)),
        int(round(xmax)),
        int(round(ymax)),
    )


# ============================================================
# XML GENERATION
# ============================================================

def create_xml(
    image_path,
    image_width,
    image_height,
    image_channels,
    objects,
):
    """
    Create Pascal VOC XML structure.
    """

    annotation = ET.Element("annotation")

    # --------------------------------------------------------
    # Folder
    # --------------------------------------------------------

    folder = ET.SubElement(
        annotation,
        "folder",
    )

    folder.text = image_path.parent.name


    # --------------------------------------------------------
    # Filename
    # --------------------------------------------------------

    filename = ET.SubElement(
        annotation,
        "filename",
    )

    filename.text = image_path.name


    # --------------------------------------------------------
    # Full path
    # --------------------------------------------------------

    path_element = ET.SubElement(
        annotation,
        "path",
    )

    path_element.text = str(
        image_path.resolve()
    )


    # --------------------------------------------------------
    # Source
    # --------------------------------------------------------

    source = ET.SubElement(
        annotation,
        "source",
    )

    database = ET.SubElement(
        source,
        "database",
    )

    database.text = "Vehicle Dataset"


    # --------------------------------------------------------
    # Image dimensions
    # --------------------------------------------------------

    size = ET.SubElement(
        annotation,
        "size",
    )

    width = ET.SubElement(
        size,
        "width",
    )

    width.text = str(
        image_width
    )

    height = ET.SubElement(
        size,
        "height",
    )

    height.text = str(
        image_height
    )

    depth = ET.SubElement(
        size,
        "depth",
    )

    depth.text = str(
        image_channels
    )


    # --------------------------------------------------------
    # Segmentation flag
    # --------------------------------------------------------

    segmented = ET.SubElement(
        annotation,
        "segmented",
    )

    segmented.text = "0"


    # --------------------------------------------------------
    # Objects
    # --------------------------------------------------------

    for obj in objects:

        object_element = ET.SubElement(
            annotation,
            "object",
        )

        # Class name
        name = ET.SubElement(
            object_element,
            "name",
        )

        name.text = obj["class_name"]


        pose = ET.SubElement(
            object_element,
            "pose",
        )

        pose.text = "Unspecified"


        truncated = ET.SubElement(
            object_element,
            "truncated",
        )

        truncated.text = "0"


        difficult = ET.SubElement(
            object_element,
            "difficult",
        )

        difficult.text = "0"


        # Bounding box
        bounding_box = ET.SubElement(
            object_element,
            "bndbox",
        )

        xmin = ET.SubElement(
            bounding_box,
            "xmin",
        )

        xmin.text = str(
            obj["xmin"]
        )

        ymin = ET.SubElement(
            bounding_box,
            "ymin",
        )

        ymin.text = str(
            obj["ymin"]
        )

        xmax = ET.SubElement(
            bounding_box,
            "xmax",
        )

        xmax.text = str(
            obj["xmax"]
        )

        ymax = ET.SubElement(
            bounding_box,
            "ymax",
        )

        ymax.text = str(
            obj["ymax"]
        )

    return annotation


# ============================================================
# PRETTY XML
# ============================================================

def indent_xml(
    element,
    level=0,
):
    """
    Add indentation so the generated XML is human-readable.
    """

    indentation = "\n" + level * "    "

    if len(element):

        if (
            not element.text
            or not element.text.strip()
        ):
            element.text = (
                indentation + "    "
            )

        for child in element:

            indent_xml(
                child,
                level + 1,
            )

        if (
            not child.tail
            or not child.tail.strip()
        ):
            child.tail = indentation

    if (
        level
        and (
            not element.tail
            or not element.tail.strip()
        )
    ):
        element.tail = indentation


# ============================================================
# CONVERT ONE IMAGE
# ============================================================

def convert_image(
    image_path,
):
    """
    Convert one YOLO label file to Pascal VOC XML.
    """

    image = cv2.imread(
        str(image_path)
    )

    if image is None:

        print(
            f"Warning: Cannot read image: "
            f"{image_path}"
        )

        return False


    image_height, image_width = (
        image.shape[:2]
    )

    if len(image.shape) == 3:

        image_channels = (
            image.shape[2]
        )

    else:

        image_channels = 1


    # --------------------------------------------------------
    # Matching YOLO label
    # --------------------------------------------------------

    label_path = (
        YOLO_LABEL_DIR
        / f"{image_path.stem}.txt"
    )


    objects = []


    # --------------------------------------------------------
    # Read YOLO labels
    # --------------------------------------------------------

    if label_path.exists():

        with label_path.open(
            "r",
            encoding="utf-8",
        ) as file:

            for line_number, line in enumerate(
                file,
                start=1,
            ):

                line = line.strip()

                if not line:
                    continue


                fields = line.split()


                if len(fields) < 5:

                    print(
                        f"Warning: Invalid annotation "
                        f"in {label_path}, "
                        f"line {line_number}"
                    )

                    continue


                try:

                    class_id = int(
                        float(fields[0])
                    )

                    x_center = float(
                        fields[1]
                    )

                    y_center = float(
                        fields[2]
                    )

                    box_width = float(
                        fields[3]
                    )

                    box_height = float(
                        fields[4]
                    )


                except ValueError:

                    print(
                        f"Warning: Invalid numeric "
                        f"annotation in "
                        f"{label_path}, "
                        f"line {line_number}"
                    )

                    continue


                # --------------------------------------------
                # Validate class ID
                # --------------------------------------------

                if (
                    class_id < 0
                    or class_id >= len(CLASS_NAMES)
                ):

                    print(
                        f"Warning: Unknown class ID "
                        f"{class_id} in "
                        f"{label_path}"
                    )

                    continue


                # --------------------------------------------
                # Convert YOLO -> absolute coordinates
                # --------------------------------------------

                (
                    xmin,
                    ymin,
                    xmax,
                    ymax,
                ) = yolo_to_voc(

                    x_center,
                    y_center,

                    box_width,
                    box_height,

                    image_width,
                    image_height,
                )


                # --------------------------------------------
                # Ignore invalid bounding boxes
                # --------------------------------------------

                if (
                    xmax <= xmin
                    or ymax <= ymin
                ):

                    print(
                        f"Warning: Invalid bounding "
                        f"box in {label_path}, "
                        f"line {line_number}"
                    )

                    continue


                objects.append({

                    "class_id":
                        class_id,

                    "class_name":
                        CLASS_NAMES[
                            class_id
                        ],

                    "xmin":
                        xmin,

                    "ymin":
                        ymin,

                    "xmax":
                        xmax,

                    "ymax":
                        ymax,
                })


    else:

        print(
            f"Notice: No label file for "
            f"{image_path.name}"
        )


    # --------------------------------------------------------
    # Generate XML
    # --------------------------------------------------------

    root = create_xml(

        image_path=image_path,

        image_width=image_width,
        image_height=image_height,

        image_channels=image_channels,

        objects=objects,
    )


    indent_xml(
        root
    )


    # --------------------------------------------------------
    # Save XML
    # --------------------------------------------------------

    output_path = (
        OUTPUT_DIR
        / f"{image_path.stem}.xml"
    )


    tree = ET.ElementTree(
        root
    )


    tree.write(
        str(output_path),
        encoding="utf-8",
        xml_declaration=True,
    )


    return True


# ============================================================
# MAIN
# ============================================================

def main():

    print(
        "=" * 70
    )

    print(
        "YOLOv8 -> Pascal VOC Conversion"
    )

    print(
        "=" * 70
    )


    # --------------------------------------------------------
    # Validate folders
    # --------------------------------------------------------

    if not IMAGE_DIR.exists():

        raise FileNotFoundError(
            f"Image directory not found: "
            f"{IMAGE_DIR.resolve()}"
        )


    if not YOLO_LABEL_DIR.exists():

        raise FileNotFoundError(
            f"YOLO label directory not found: "
            f"{YOLO_LABEL_DIR.resolve()}"
        )


    OUTPUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )


    # --------------------------------------------------------
    # Find images
    # --------------------------------------------------------

    images = sorted(

        [
            path
            for path in IMAGE_DIR.iterdir()

            if (
                path.is_file()
                and path.suffix.lower()
                in IMAGE_EXTENSIONS
            )
        ]

    )


    if not images:

        raise RuntimeError(
            f"No images found in "
            f"{IMAGE_DIR.resolve()}"
        )


    print(
        f"Images: {len(images)}"
    )

    print(
        f"Input images: "
        f"{IMAGE_DIR.resolve()}"
    )

    print(
        f"YOLO labels: "
        f"{YOLO_LABEL_DIR.resolve()}"
    )

    print(
        f"VOC output: "
        f"{OUTPUT_DIR.resolve()}"
    )

    print()


    # --------------------------------------------------------
    # Convert
    # --------------------------------------------------------

    successful = 0


    for index, image_path in enumerate(
        images,
        start=1,
    ):

        success = convert_image(
            image_path
        )

        if success:
            successful += 1


        if (
            index % 100 == 0
            or index == len(images)
        ):

            print(
                f"Processed "
                f"{index}/{len(images)}"
            )


    # --------------------------------------------------------
    # Summary
    # --------------------------------------------------------

    print()

    print(
        "=" * 70
    )

    print(
        "Conversion Complete"
    )

    print(
        "=" * 70
    )

    print(
        f"Images processed : "
        f"{len(images)}"
    )

    print(
        f"XML files created: "
        f"{successful}"
    )

    print(
        f"Output directory : "
        f"{OUTPUT_DIR.resolve()}"
    )


if __name__ == "__main__":
    main()
