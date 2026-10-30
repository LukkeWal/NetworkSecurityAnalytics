"""
Shows that the HIKARI-2021 CSV lists many attack flows twice, as the attack and as Benign or Background with identical features.
"""

import pandas as pd

import DataHandler

FLOW_KEY = ["originh", "originp", "responh", "responp", "flow_duration", "fwd_pkts_tot", "bwd_pkts_tot"]


def main() -> None:
    data = DataHandler.read_data()
    features = [column for column in data.select_dtypes(include="number").columns if column not in FLOW_KEY + ["Label"]]
    others = data[data["traffic_category"].isin(["Benign", "Background"])]

    print(f"{len(data)} rows, {data['uid'].nunique()} distinct uids")
    for category in ["Probing", "Bruteforce-XML", "Bruteforce", "XMRIGCC CryptoMiner"]:
        attack = data[data["traffic_category"] == category]
        twins = attack.merge(others, on=FLOW_KEY, suffixes=("", "_twin"))
        identical = (twins[features].to_numpy() == twins[[f"{column}_twin" for column in features]].to_numpy()).all(axis=1)
        print(f"{category:20s} {len(attack):6d} rows, {twins[FLOW_KEY].drop_duplicates().shape[0]:6d} with a twin, "
              f"{identical.mean() if len(twins) else 0:.1%} of twins identical in all {len(features)} other numeric features")

    split = DataHandler.get_stratified_split(data.assign(row_id=range(len(data))))
    split_of = {row: name for name, part in [("train", split.train), ("val", split.val), ("test", split.test)]
                for row in part.x["row_id"]}
    labelled = data.assign(split=pd.Series(split_of))
    probing = labelled[labelled["traffic_category"] == "Probing"].merge(
        labelled[labelled["traffic_category"] == "Benign"], on=FLOW_KEY, suffixes=("", "_twin")
    )
    print("\nProbing row split (rows) against the split of its Benign twin (columns):")
    print(pd.crosstab(probing["split"], probing["split_twin"]))


if __name__ == "__main__":
    main()
