import argparse
import json
import random
from collections import defaultdict
from pathlib import Path


def sample_routes(input_path: Path, per_destination: int, seed: int | None) -> Path:
    with input_path.open("r", encoding="utf-8") as infile:
        data = json.load(infile)

    routes = data.get("routes", [])
    grouped_routes: dict[str, list[dict]] = defaultdict(list)

    for route in routes:
        if "destinationPolygonKey" not in route:
            raise ValueError("Route is missing destinationPolygonKey.")
        grouped_routes[route["destinationPolygonKey"]].append(route)

    rng = random.Random(seed)
    sampled_routes: list[dict] = []

    for key, routes_for_dest in grouped_routes.items():
        if per_destination >= len(routes_for_dest):
            sampled_routes.extend(routes_for_dest)
        else:
            sampled_routes.extend(rng.sample(routes_for_dest, per_destination))

    output_data = dict(data)
    output_data["routes"] = sampled_routes

    output_path = input_path.with_name(f"{input_path.stem}_{per_destination}{input_path.suffix}")
    with output_path.open("w", encoding="utf-8") as outfile:
        json.dump(output_data, outfile, indent=2)
        outfile.write("\n")

    return output_path


def main() -> None:
    parser = argparse.ArgumentParser(description="Sample N routes per destination from a paths JSON file.")
    parser.add_argument("paths_file", type=Path, help="Path to the input routes JSON file.")
    parser.add_argument("per_destination", type=int, help="Number of routes to sample per destination.")
    parser.add_argument("--seed", type=int, default=None, help="Optional random seed for reproducible sampling.")

    args = parser.parse_args()
    if args.per_destination <= 0:
        raise ValueError("per_destination must be a positive integer.")

    output_path = sample_routes(args.paths_file, args.per_destination, args.seed)
    print(output_path)


if __name__ == "__main__":
    main()

