class DataDescriber:
    """
    Class object to get information about a spark dataframe.

    Methods
    ::sample_df: retrieves n-samples of the data.
    ::stats_df: calculates descriptive statistics for numeric column values.

    Parameters
    ::param spark_df:  source data_frame.
    ::param target_sample_size:  number of samples to include for each column.
    ::return spark_df: returns a dataframe whose
    """

    def __init__(self, spark_df: SPARK_DF, sample_count: int = 3):
        self.spark_df = spark_df
        self.describe_df = None
        self.sample_count = sample_count
        self.row_count = spark_df.count()
        self.sample_pct = self._calculate_sample_pct()
        self.schema_json = self._spark_schema_to_json()
        self.describe_df_column_names = {
            "index": "COLUMN_NAMES"
        }
        self.null_df_column_names = {
            "index": "COLUMN_NAMES",
            0: "NULL_COUNT"
        }
        self.levels_df_column_names = {
            "index": "COLUMN_NAMES",
            0: "LEVEL_COUNT",
            "PRIMARY_KEY": "PRIMARY_KEY"
        }
        self.stats_df_column_names = {
            'index': 'COLUMN_NAMES',
            1: 'STAT_MEAN',
            2: 'STAT_STDDEV',
            3: 'STAT_MIN',
            4: 'STAT_PCTL_25',
            5: 'STAT_PCTL_50',
            6: 'STAT_PCTL_75',
            7: 'STAT_MAX'
        }
        self.describe_df_schema = {
            "DATA_TYPE": {'type_name': 'string', 'type': str},
            "LEVEL_COUNT": {'type_name': 'integer', 'type': int},
            "UNIQUE_KEY": {'type_name': 'string', 'type': str},
            "NULL_COUNT": {'type_name': 'integer', type: int},
            "PCT_NULL": {'type_name': 'float', 'type': float},
            "STAT_MEAN": {'type_name': 'float', 'type': float},
            'STAT_STDDEV': {'type_name': 'float', 'type': float},
            'STAT_MIN': {'type_name': 'float', 'type': float},
            'STAT_PCTL_25': {'type_name': 'float', 'type': float},
            'STAT_PCTL_50': {'type_name': 'float', 'type': float},
            'STAT_PCTL_75': {'type_name': 'float', 'type': float},
            'STAT_MAX': {'type_name': 'float', 'type': float},
        }
        logger.info("Data Dictionary class successfully instantiated")

    def _calculate_sample_pct(self):
        return self.sample_count / self.spark_df.count()

    def _get_sample_df_renamed_cols(self, pd_df: pd.DataFrame):
        """
        Construct dictionary to rename columns of a pandas DataFrame.
        Position 0 is the index of column names.
        Position 1->n are samples
        """
        colnames = {c: f"SAMPLE_{c}" for c in pd_df.columns[1:]}
        colnames.update({'index': 'COLUMN_NAMES'})
        return colnames

    def _spark_schema_to_json(self):
        schema_json = {
            x['name']: x['type'] for x in self.spark_df.schema.jsonValue()['fields']
        }
        return schema_json

    def _get_non_str_colnames(self):
        columns = [
            x for x in self.spark_df.columns
            if self.schema_json[x] not in ('string')
        ]
        return columns

    def _get_categorical_columns(self):
        columns = [
            x for x in self.spark_df.columns
            if self.schema_json[x] in ('string')
        ]
        return columns

    def _get_describe_df(self):
        spark_columns = (
            self
            .spark_df
            .columns
        )
        spark_schema = (
            self
            ._spark_schema_to_json()
        )
        self.describe_df = (
            pd
            .DataFrame(index=spark_columns)
            .reset_index()
            .rename(columns=self.describe_df_column_names)
        )
        # Add Schema Column
        self.describe_df['DATA_TYPE'] = [spark_schema[c] for c in spark_columns]

        return self.describe_df

    def sample_df(self):
        """
        """
        df_sample = (
            self.spark_df  # noqa
            .select(
                *(col for col in self.spark_df.columns
                  if (
                      self.spark_df
                      .where(F.col(col).isNull())
                      .count() < self.row_count
                  )
                )
            )
            .sample(fraction=self.sample_pct)
            .toPandas()
            .head(self.sample_count)
            .transpose()
            .reset_index()
        )

        df_sample = (
            df_sample
            .rename(columns=self._get_sample_df_renamed_cols(df_sample))
            .fillna("")
            .astype(str)
        )

        return df_sample

    def stats_df(self):
        """
        """
        stats_df = (
            self.spark_df
            .select(self._get_non_str_colnames())
            .summary()
            .toPandas()
            .fillna(0.0)
            .transpose()
            .reset_index()
            .drop(0, axis=0)  # drop column names
            .rename(columns=self.stats_df_column_names)
            .astype({c: float for c in list(self.stats_df_column_names.values())[1:]})
            .round(3)
            .drop(0, axis=1)  # drop count column
        )
        return stats_df

    def levels_df(self):
        """
        Identify number of levels per categorical column.
        """
        cat_df = (
            self.spark_df
            .select(
                [
                    F.countDistinct(c).alias(c)
                    for c in self._get_categorical_columns()
                ]
            )
            .toPandas()
            .transpose()
            .reset_index()
            .rename(columns=self.levels_df_column_names)
        )

        # Create Unique Key Column
        cat_df["UNQIUE_KEY"] = list(
            map(
                lambda x: 'True' if x == self.row_count else "",
                cat_df.LEVEL_COUNT
            )
        )

        return cat_df

    def null_df(self):
        """
        """
        null_df = (
            self.spark_df
            .select(
                [
                    F.count(
                        F.when(
                            (
                                F.col(c).isNull() |
                                F.col(c).contains('None') |
                                F.col(c).contains('NULL')
                            ), c)
                        ).alias(c)
                    for c in self.spark_df.columns
                ]
            )
            .toPandas()
            .fillna(0)
            .astype(int)
            .transpose()
            .reset_index()
            .rename(columns=self.null_df_column_names)
        )

        null_df["PCT_NULL"] = (
            null_df
            .NULL_COUNT / self.spark_df.count()
        ).round(2)

        return null_df

    def describe(self):
        """
        Create describe dataframe.
        """
        self.describe_df = self._get_describe_df()
        frames = [
            self.sample_df(),
            self.levels_df(),
            self.null_df(),
            self.stats_df(),
        ]

        for df in frames:

            self.describe_df = (
                self.describe_df
                .merge(
                    df,
                    on=["COLUMN_NAMES"],
                    how='left'
                )
            )

        return self

    def _assert_schema(self):

        # Convert Everything to String
        self.describe_df = self.describe_df.fillna('').astype(str)

        # Convert Numerica Columns to Respective Dtype
        for c in self.describe_df_schema.keys():

            if self.describe_df_schema[c]['type_name'] == 'float':
                self.describe_df[c] = list(map(lambda x: np.nan if not x else x, self.describe_df[c]))
                self.describe_df = self.describe_df.astype({c: float})

            if self.describe_df_schema[c]['type_name'] == 'integer':
                self.describe_df[c] = list(map(lambda x: 0 if not x else x, self.describe_df[c]))
                self.describe_df[c] = list(map(lambda x: int(str(x).split('.')[0]), self.describe_df[c]))

        return self

    def convert_pandas_df_to_spark_df(self, ctx: SparkSession):
        self._assert_schema()
        return ctx.spark_session.createDataFrame(self.describe_df)
