import os

from t_tech.invest import Client
from t_tech.invest.utils import money_to_decimal


TOKEN = os.environ["INVEST_TOKEN"]


def main():
    """Entry point: print account and portfolio totals."""
    with Client(TOKEN) as client:
        accounts_response = client.users.get_accounts()
        if not accounts_response.accounts:
            print("No available accounts.")
            return
        account_id = accounts_response.accounts[0].id
        print(
            f"Using account: \
            {accounts_response.accounts[0].name} \
                (ID: {account_id})"
        )
        portfolio = client.operations.get_portfolio(account_id=account_id)
        total = portfolio.total_amount_portfolio
        print(
            f"Total portfolio balance: "
            f"{money_to_decimal(total)} {total.currency}"
        )
        print("\nMoney positions:")
        for money in portfolio.positions:
            if money.instrument_type == "currency":
                print(
                    f" {money.instrument_type}: "
                    f"{money.balance} {money.currency}"
                )


if __name__ == "__main__":
    main()
