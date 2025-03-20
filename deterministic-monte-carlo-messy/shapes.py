import random
import matplotlib as plt

def generate_shapes_no_overlap_no_cutoff(seed=None, num_shapes=10, canvas_size=(10, 10), save_path=None):
    """
    Generate a random set of shapes with varying colors, ensuring no overlap and no shapes cut off at edges.

    Parameters:
        seed (int): Random seed for reproducibility.
        num_shapes (int): Number of shapes to generate.
        canvas_size (tuple): Dimensions of the canvas (width, height).
        save_path (str): Path to save the generated image. If None, image will not be saved.

    Returns:
        None
    """
    if seed is not None:
        random.seed(seed)

    fig, ax = plt.subplots(figsize=(canvas_size[0], canvas_size[1]))

    existing_shapes = []

    def check_validity(new_shape_bbox):
        """Check if a new shape is valid (no overlap and not cut off)."""
        # Check if the shape is within the canvas
        if new_shape_bbox[0] < 0 or new_shape_bbox[1] < 0 or \
           new_shape_bbox[2] > canvas_size[0] or new_shape_bbox[3] > canvas_size[1]:
            return False

        # Check if the shape overlaps with existing shapes
        for existing_bbox in existing_shapes:
            if (new_shape_bbox[0] < existing_bbox[2] and  # Left < Existing Right
                new_shape_bbox[2] > existing_bbox[0] and  # Right > Existing Left
                new_shape_bbox[1] < existing_bbox[3] and  # Bottom < Existing Top
                new_shape_bbox[3] > existing_bbox[1]):    # Top > Existing Bottom
                return False

        return True

    def add_shape(shape, bbox):
        """Add shape and its bounding box to the canvas and list."""
        existing_shapes.append(bbox)
        ax.add_patch(shape)

    colors = ['red', 'blue', 'green', 'purple', 'orange', 'yellow', 'pink', 'cyan']

    while len(existing_shapes) < num_shapes:
        x, y = random.uniform(0, canvas_size[0]), random.uniform(0, canvas_size[1])
        w, h = random.uniform(1, 3), random.uniform(1, 3)
        shape_type = random.choice(["circle", "rectangle", "ellipse"])

        if shape_type == "circle":
            radius = w / 2
            bbox = (x - radius, y - radius, x + radius, y + radius)
            if check_validity(bbox):
                shape = patches.Circle((x, y), radius, facecolor=random.choice(colors), edgecolor="black")
                add_shape(shape, bbox)

        elif shape_type == "rectangle":
            bbox = (x, y, x + w, y + h)
            if check_validity(bbox):
                shape = patches.Rectangle((x, y), w, h, facecolor=random.choice(colors), edgecolor="black")
                add_shape(shape, bbox)

        elif shape_type == "ellipse":
            bbox = (x - w / 2, y - h / 2, x + w / 2, y + h / 2)
            if check_validity(bbox):
                shape = patches.Ellipse((x, y), w, h, facecolor=random.choice(colors), edgecolor="black")
                add_shape(shape, bbox)

    ax.set_xlim(0, canvas_size[0])
    ax.set_ylim(0, canvas_size[1])
    ax.set_aspect('equal')
    plt.axis('off')

    if save_path:
        plt.savefig(save_path, dpi=64, bbox_inches='tight', pad_inches=0)
    plt.show()

# Generate non-overlapping shapes without edge cutoff
output_path_no_cutoff = "/mnt/data/random_shapes_no_cutoff.png"
generate_shapes_no_overlap_no_cutoff(seed=42, num_shapes=15, canvas_size=(10, 10), save_path=output_path_no_cutoff)
