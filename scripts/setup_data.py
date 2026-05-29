"""This pulls all the data needed for the example. 

The ideas is thatyou will run once after installing dependencies. 

Each step stores its output under
data/cache, so re-running isn't goinf to take too long, hopefully making this snappy

    uv run python scripts/setup_data.py
"""

from geokg import boundaries, heat, index


def main() -> None:
    print("1/3  Swiss City boundaries being pulled from swissBOUNDARIES3D")
    cities = boundaries.build_city_boundaries()
    print(f"     {len(cities)} cities ready.")

    print("2/3  Surface temperature series (Coming directly from Landsat, so this can take a few minutes)")
    df = heat.build_heat_series()
    print(f"     {len(df)} city-year scenes cached.")

    print("3/3  Vector index (fastembed downloads a small model on first run)")
    n = index.build_index()
    print(f"     {n} document chunks indexed.")

    print("Done. To launch the dashboard, run :  uv run streamlit run app/dashboard.py")


if __name__ == "__main__":
    main()
