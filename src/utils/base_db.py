from pandas import pd


class SearchEngine:
    def __init__(self, data: list[str]):
        self.data = data
        self.engine = None
        self.index_data()

    def index_data(self) -> None:
        raise NotImplemented

    def search(self, query: str, k: int = 10) -> pd.DataFrame:
        raise NotImplemented
