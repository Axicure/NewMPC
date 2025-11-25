import pandas as pd
import matplotlib.pyplot as plt


def plot_resource_contention(
    mpc_csv: str = "mpc_results_detailed.csv",
    random_csv: str = "random_results_detailed.csv",
) -> None:
    """Plot resource_contention vs window_id for MPC vs random strategies."""
    mpc_df = pd.read_csv(mpc_csv)
    random_df = pd.read_csv(random_csv)

    required_cols = {"window_id", "resource_contention"}
    for name, df in (("MPC", mpc_df), ("Random", random_df)):
        if not required_cols.issubset(df.columns):
            missing = required_cols - set(df.columns)
            raise ValueError(f"{name} CSV missing columns: {missing}")

    plt.figure(figsize=(12, 6))
    plt.plot(
        mpc_df["window_id"],
        mpc_df["resource_contention"],
        label="MPC",
        linewidth=2,
    )
    plt.plot(
        random_df["window_id"],
        random_df["resource_contention"],
        label="Random",
        linewidth=2,
        linestyle="--",
    )
    plt.xlabel("Window ID")
    plt.ylabel("Resource Contention")
    plt.title("Resource Contention Over Time Windows")
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    plot_resource_contention()

