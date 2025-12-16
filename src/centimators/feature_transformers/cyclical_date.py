"""Cyclical date feature transformer for temporal pattern encoding."""

import narwhals as nw
import numpy as np
from narwhals.typing import FrameT, IntoSeries

from .base import _BaseFeatureTransformer


class CyclicalDateTransformer(_BaseFeatureTransformer):
    """
    CyclicalDateTransformer encodes date components as cyclical features using sine and cosine.

    This transformer extracts various time components (month, week, day, etc.) from a date
    column and applies trigonometric encoding to capture their cyclical nature. This helps
    machine learning models learn temporal patterns like seasonality, weekly cycles, etc.

    The cyclical encoding formula:
        sin_value = sin(2 * π * value / period)
        cos_value = cos(2 * π * value / period)

    Args:
        components (list of str, optional): Time components to encode.
            Options: 'month', 'week', 'day_of_year', 'day_of_week', 'day_of_month'
            If None, all components are used.
        prefix (str, optional): Prefix for output column names. Defaults to 'feature_'.
        feature_names (list of str, optional): Not used for this transformer, kept for compatibility.

    Examples:
        >>> import pandas as pd
        >>> from centimators.feature_transformers import CyclicalDateTransformer
        >>> df = pd.DataFrame({
        ...     'date': pd.date_range('2021-01-01', periods=5, freq='D')
        ... })
        >>> transformer = CyclicalDateTransformer(components=['month', 'day_of_week'])
        >>> result = transformer.fit_transform(df, date_series=df['date'])
        >>> print(result.columns)
        Index(['feature_month_sin', 'feature_month_cos',
               'feature_day_of_week_sin', 'feature_day_of_week_cos'], dtype='object')
    """

    # Period for each component (max value for normalization)
    COMPONENT_PERIODS = {
        'month': 12,
        'week': 52,
        'day_of_year': 365,
        'day_of_week': 7,
        'day_of_month': 31
    }

    def __init__(
        self,
        components=None,
        prefix='feature_',
        feature_names=None
    ):
        """Initialize the CyclicalDateTransformer.

        Args:
            components: List of time components to encode. If None, uses all components.
            prefix: Prefix for output column names.
            feature_names: Not used, kept for compatibility with base class.
        """
        self.components = components or list(self.COMPONENT_PERIODS.keys())
        self.prefix = prefix
        super().__init__(feature_names=None)  # This transformer doesn't use feature_names

        # Validate components
        invalid = set(self.components) - set(self.COMPONENT_PERIODS.keys())
        if invalid:
            raise ValueError(
                f"Invalid components: {invalid}. "
                f"Valid options: {list(self.COMPONENT_PERIODS.keys())}"
            )

    @nw.narwhalify(allow_series=True)
    def transform(
        self,
        X: FrameT,
        y=None,
        date_series: IntoSeries = None,
    ) -> FrameT:
        """Apply cyclical date encoding.

        Args:
            X (FrameT): Input data frame (not used, but kept for sklearn compatibility).
            y (Any, optional): Ignored. Kept for compatibility.
            date_series (IntoSeries): Series containing date values to encode.

        Returns:
            FrameT: DataFrame with cyclical encoded date features.

        Raises:
            ValueError: If date_series is not provided.
        """
        if date_series is None:
            raise ValueError(
                "date_series must be provided for CyclicalDateTransformer. "
                "Use .set_transform_request(date_series=True) when using in a pipeline."
            )

        # Create a temporary dataframe with the date series
        # We need to convert the series to a dataframe to work with it
        date_native = nw.to_native(date_series)

        # Create a dataframe from the series with proper naming
        if hasattr(date_native, 'to_frame'):
            # Polars Series
            temp_df = nw.from_native(date_native.to_frame('date_col'))
        else:
            # Pandas Series
            import pandas as pd
            temp_df = nw.from_native(pd.DataFrame({'date_col': date_native}))

        # Extract date components first
        date_components = {}

        for component in self.components:
            # Extract the date component using narwhals expressions
            if component == 'month':
                temp_df = temp_df.with_columns(
                    nw.col('date_col').dt.month().alias(f'{component}_value')
                )
            elif component == 'week':
                # Narwhals doesn't have week() method, so we'll calculate it from ordinal_day
                # Week number = (ordinal_day - 1) // 7 + 1 (ISO week approximation)
                temp_df = temp_df.with_columns(
                    ((nw.col('date_col').dt.ordinal_day() - 1) // 7 + 1).alias(f'{component}_value')
                )
            elif component == 'day_of_year':
                temp_df = temp_df.with_columns(
                    nw.col('date_col').dt.ordinal_day().alias(f'{component}_value')
                )
            elif component == 'day_of_week':
                temp_df = temp_df.with_columns(
                    nw.col('date_col').dt.weekday().alias(f'{component}_value')
                )
            elif component == 'day_of_month':
                temp_df = temp_df.with_columns(
                    nw.col('date_col').dt.day().alias(f'{component}_value')
                )

            date_components[component] = f'{component}_value'

        # Now apply sin/cos transformations using numpy
        # Get the native dataframe to add array columns
        temp_native = nw.to_native(temp_df)
        result_columns = []

        for component in self.components:
            period = self.COMPONENT_PERIODS[component]
            col_name = date_components[component]

            # Get the values as numpy array
            values = nw.to_native(temp_df[col_name])

            # Convert to numpy if needed
            if hasattr(values, 'to_numpy'):
                values = values.to_numpy()
            elif hasattr(values, 'values'):
                values = values.values
            else:
                values = np.array(values)

            # Apply cyclical encoding: 2 * π * value / period
            angle = 2 * np.pi * values / period

            # Calculate sin and cos
            sin_values = np.sin(angle)
            cos_values = np.cos(angle)

            # Add columns to native dataframe
            sin_col_name = f"{self.prefix}{component}_sin"
            cos_col_name = f"{self.prefix}{component}_cos"

            # Handle both polars and pandas
            if hasattr(temp_native, 'with_columns'):
                # Polars
                import polars as pl
                temp_native = temp_native.with_columns([
                    pl.Series(sin_col_name, sin_values),
                    pl.Series(cos_col_name, cos_values)
                ])
            else:
                # Pandas
                temp_native[sin_col_name] = sin_values
                temp_native[cos_col_name] = cos_values

            result_columns.extend([sin_col_name, cos_col_name])

        # Convert back to narwhals and select only the result columns
        temp_df = nw.from_native(temp_native)
        result_df = temp_df.select(result_columns)

        # Store output names for get_feature_names_out
        self._output_names = result_df.columns

        return result_df

    def get_feature_names_out(self, input_features=None) -> list[str]:
        """Return output feature names.

        Args:
            input_features (list[str], optional): Ignored. Kept for compatibility.

        Returns:
            list[str]: List of transformed feature names.
        """
        if hasattr(self, '_output_names'):
            return self._output_names

        # Generate names if transform hasn't been called yet
        names = []
        for component in self.components:
            names.append(f"{self.prefix}{component}_sin")
            names.append(f"{self.prefix}{component}_cos")
        return names
