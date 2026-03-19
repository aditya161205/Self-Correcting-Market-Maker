from dataclasses import dataclass


@dataclass
class Portfolio:
    inventory: int = 0
    cash: float = 0.0

    def update_on_bid_fill(self, bid_price: float) -> None:
        """
        Bid fill => we buy 1 unit
        """
        self.inventory += 1
        self.cash -= bid_price

    def update_on_ask_fill(self, ask_price: float) -> None:
        """
        Ask fill => we sell 1 unit
        """
        self.inventory -= 1
        self.cash += ask_price

    def wealth(self, mid_price: float) -> float:
        """
        Mark-to-market wealth
        """
        return self.cash + self.inventory * mid_price