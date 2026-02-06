class SymbolIndexer:
    """
    Bi-directional mapping:
        symbol -> id
        id -> symbol
    """

    def __init__(self, symbols):
        self.symbol_to_idx = {s: i for i, s in enumerate(sorted(symbols))}
        self.idx_to_symbol = {i: s for s, i in self.symbol_to_idx.items()}

    def symbol_to_id(self, symbol: str) -> int:
        return self.symbol_to_idx[symbol]

    def id_to_symbol(self, idx: int) -> str:
        return self.idx_to_symbol[idx]

    def __len__(self):
        return len(self.symbol_to_idx)
