import pandas as pd

from api.utils.logger import logger


class CSVLoader:
    def __init__(self, db_engine):
        self.db_engine = db_engine

    def load(self, file_url: str, table_name: str):
        try:
            df = pd.read_csv(file_url)
            # Clean the column title format to make it easy for DB query
            df.columns = map(lambda x: "_".join(x.lower().split()), df.columns)
            df.to_sql(
                table_name, con=self.db_engine, if_exists="replace", index=False
            )
            logger.info(f"Successfully loaded csv file {file_url} into DB")
        except Exception as e:
            logger.exception(f"Failed to load csv file {file_url} into DB: {e}")
