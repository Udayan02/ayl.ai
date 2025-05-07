# ayl.ai
We're an edge ai, IoT and embedded vision company aiming to transform the in-store shopping experience through intelligent automation and operations research

# Demo Day 1 update:
## For plotting only the store (no Dijkstra's)
```python store_pathfinder.py tag_traverse --graph_file ./final_store_graph.pkl --json_file ./filtered_output_with_images.json --tags "Nike" --output_file ./output/overall_map.png --hide_labels --show_images --image_dir demo_images --image_size 0.05```<br>
Note: Replace "Nike" with a tag that is **not** present in the store.
## For plotting a Dijkstra's optimal path for a given set of items
```python store_pathfinder.py tag_traverse --graph_file ./final_store_graph.pkl --json_file ./filtered_output_with_images.json --tags "Cashew Nuts" "Skinny Pop" "Celsius" --output_file ./output/dijkstra_output.png --hide_labels --show_images --image_dir demo_images --image_size 0.05```<br>
Note: There are some issues to be ironed out for specific tags.
