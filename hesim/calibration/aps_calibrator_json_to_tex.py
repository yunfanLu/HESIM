import argparse
import json

import pandas as pd


def main(json_file, out_csv_file):
    with open(json_file, "r") as f:
        data = json.load(f)

    term_names = [f"Mu^{t['P']} * T^{t['O']}" for t in data["terms"]]

    rows = []
    for key, pixel_terms in data["betas"].items():
        row = {"Pos": key}
        for tname in term_names:
            coeff = None
            for term in pixel_terms:
                if term["term"] == tname:
                    coeff = term["coefficient"]
                    break
            row[tname] = coeff
        rows.append(row)

    df = pd.DataFrame(rows)
    df = df.sort_values("Pos").reset_index(drop=True)
    print(df)
    df.to_csv(out_csv_file, index=False)
    with open(out_csv_file.replace(".csv", ".tex"), "w") as f:
        f.write(df.to_latex(index=False, float_format="%.6e"))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Export fitted APS noise coefficients to CSV and LaTeX.")
    parser.add_argument("json_file")
    parser.add_argument("out_csv_file")
    args = parser.parse_args()
    main(args.json_file, args.out_csv_file)
