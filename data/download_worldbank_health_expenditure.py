import requests
import pandas as pd
from pathlib import Path


def download_worldbank_health_expenditure(
    indicator="SH.XPD.CHEX.PC.CD",
    out_path="data/raw/health_expenditure_all_countries.csv",
):
    """
    Downloads Current health expenditure per capita (in USD)
    from the World Bank API for ALL countries, all available years,
    and saves a long-format CSV:

        country,iso3,year,value

    Indicator default:
        SH.XPD.CHEX.PC.CD = Current health expenditure per capita (US$)
    """

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    base_url = "https://api.worldbank.org/v2/country/all/indicator/{}".format(
        indicator
    )

    # First request to get number of pages
    params = {"format": "json", "per_page": 20000, "page": 1}
    r = requests.get(base_url, params=params)
    r.raise_for_status()
    data = r.json()

    if not isinstance(data, list) or len(data) < 2:
        raise RuntimeError(f"Unexpected response format: {data}")

    meta, records = data
    pages = meta.get("pages", 1)

    all_rows = []

    def process_records(records):
        for rec in records:
            country = rec.get("country", {}).get("value")
            iso3 = rec.get("countryiso3code")
            year = rec.get("date")
            value = rec.get("value")

            if country is None or iso3 is None or year is None:
                continue

            # Only keep rows with a numeric value
            if value is None:
                continue

            all_rows.append(
                {
                    "country": country,
                    "iso3": iso3,
                    "year": int(year),
                    "value": float(value),
                }
            )

    # Process first page
    process_records(records)

    # Process remaining pages (if any)
    for page in range(2, pages + 1):
        params["page"] = page
        r = requests.get(base_url, params=params)
        r.raise_for_status()
        data = r.json()
        if not isinstance(data, list) or len(data) < 2:
            continue
        _, records = data
        process_records(records)

    if not all_rows:
        raise RuntimeError("No data downloaded from World Bank API.")

    df = pd.DataFrame(all_rows)
    df = df.sort_values(["iso3", "year"]).reset_index(drop=True)

    # Save long-format CSV
    df.to_csv(out_path, index=False)
    print(f"Saved {len(df)} rows to {out_path}")


if __name__ == "__main__":
    download_worldbank_health_expenditure()

