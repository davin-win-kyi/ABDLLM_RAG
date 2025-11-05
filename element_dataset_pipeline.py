# element_dataset_pipeline.py
from pathlib import Path
from initialize_element_dataset import build_initial_dataset
from element_neighbors import add_neighbors
from element_feedback_loop import feedback_loop

def main():
    html = Path("reviewable_page.html")
    out = Path("element_dataset.json")

    # 1) Initialize dataset (IDs, types, base descriptors via Selenium)
    build_initial_dataset(html, out)

    # 2) Find neighbors spatially and add their DESCRIPTORS
    add_neighbors(html, out, k=5)

    # 3) Collect interactive feedback in terminal and persist
    #    (Press Enter or 'no' to skip any given element.)
    feedback_loop(html, out)

if __name__ == "__main__":
    main()
