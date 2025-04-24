import streamlit as st
import pandas as pd
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
import io
from scipy.stats import spearmanr, pearsonr
from statsmodels.stats.multitest import multipletests
import numpy as np
from scipy import stats
import matplotlib.pyplot as plt
from scipy.stats import chi2_contingency
from statsmodels.stats.multitest import multipletests
from statsmodels.graphics.mosaicplot import mosaic
import matplotlib.pyplot as plt


# Initialize session state for persistence
if 'df' not in st.session_state:
    st.session_state.df = None
if 'signal_results' not in st.session_state:
    st.session_state.signal_results = None
if 'has_results' not in st.session_state:
    st.session_state.has_results = False
if 'breakdown_results' not in st.session_state:
    st.session_state.breakdown_results = None
if 'selected_breakdown_signal' not in st.session_state:
    st.session_state.selected_breakdown_signal = None
if 'control_df' not in st.session_state:
    st.session_state.control_df = None
if 'variation_df' not in st.session_state:
    st.session_state.variation_df = None
if 'comparative_results' not in st.session_state:
    st.session_state.comparative_results = None
if 'has_comparative_results' not in st.session_state:
    st.session_state.has_comparative_results = False

st.title("The BoGo Mindset")
tab1, tab2 = st.tabs(["Single Dataset Analysis", "Comparative Analysis"])
st.subheader("An app for correlation analyses between signals and business outcomes")
st.write("Upload a file to analyze the relationship between a signal column and a business metric.")
st.write("Examples of signals could be: an eval metric, a label like Urgent/Non-urgent, the character count of a user's first message, the theme label of the chat, etc.")

with tab1:
    # Function to identify outliers
    def identify_outliers(data, column):
        """
        Identify outliers in a column using the IQR method
        Returns indices of outliers
        """
        Q1 = np.percentile(data[column], 25)
        Q3 = np.percentile(data[column], 75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR
        
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers.index

    # Function to calculate correlation confidence intervals
    def correlation_ci(r, n, alpha=0.05):
        """
        Calculate confidence interval for correlation coefficient
        """
        z = np.arctanh(r)  # Fisher's z-transformation
        se = 1/np.sqrt(n-3)
        z_crit = stats.norm.ppf(1-alpha/2)  # Critical value for the specified alpha
        
        ci_lower = np.tanh(z - z_crit*se)
        ci_upper = np.tanh(z + z_crit*se)
        
        return ci_lower, ci_upper

    def run_chi_square_test(signal_data, metric_data):
        """
        Runs a chi-square test for independence using a contingency table.
        
        Parameters:
            signal_data (pd.Series): Categorical signal data.
            metric_data (pd.Series): Categorical metric data.
            
        Returns:
            contingency_table (pd.DataFrame): Observed counts.
            chi2 (float): Chi-square test statistic.
            p (float): p-value.
            dof (int): Degrees of freedom.
            expected (np.ndarray): Expected counts.
        """
        contingency_table = pd.crosstab(signal_data, metric_data)
        chi2, p, dof, expected = stats.chi2_contingency(contingency_table)
        return contingency_table, chi2, p, dof, expected

    def plot_chi_square_heatmap(contingency_table):
        """
        Plots a heatmap for the given contingency table.
        
        Parameters:
            contingency_table (pd.DataFrame): Observed counts from the chi-square test.
        """
        fig, ax = plt.subplots()
        cax = ax.matshow(contingency_table, cmap='viridis')
        fig.colorbar(cax)
        
        # Set ticks and labels
        ax.set_xticks(np.arange(len(contingency_table.columns)))
        ax.set_xticklabels(contingency_table.columns, rotation=45, ha='right')
        ax.set_yticks(np.arange(len(contingency_table.index)))
        ax.set_yticklabels(contingency_table.index)
        
        # Annotate each cell with the count
        for (i, j), val in np.ndenumerate(contingency_table.values):
            ax.text(j, i, f'{val}', ha='center', va='center', color='white')
        
        st.pyplot(fig)

    uploaded_file = st.file_uploader("Choose a file", type=["csv", "xlsx", "xls"])

    if uploaded_file is not None:
        # Load data based on file type
        file_extension = uploaded_file.name.split('.')[-1].lower()
        
        try:
            if file_extension in ['xlsx', 'xls']:
                df = pd.read_excel(uploaded_file)
            else:  # csv
                df = pd.read_csv(uploaded_file)
            
            st.session_state.df = df
            
            # Display the dataframe
            st.write("Data Preview:")
            st.dataframe(df.head())

            # Column selection (OUTSIDE THE FORM)
            columns = df.columns.tolist()
            signal_column = st.selectbox("Select the signal column (e.g., 'Complexity' or 'Personalization')", columns, key="signal_col_select")
            metric_column = st.selectbox("Select the business metric column (e.g., 'Conversion')", columns, key="metric_col_select")
            
            # Handle NaN and Inf values only in selected columns
            original_row_count = len(df)
            
            # Create a copy of df with clean values for analysis
            analysis_df = df.copy()
            
            # Replace inf with NaN - only for numeric columns
            mask_inf = pd.Series(np.zeros(len(analysis_df), dtype=bool))
            if pd.api.types.is_numeric_dtype(analysis_df[signal_column]):
                mask_inf = mask_inf | np.isinf(analysis_df[signal_column])
            if pd.api.types.is_numeric_dtype(analysis_df[metric_column]):
                mask_inf = mask_inf | np.isinf(analysis_df[metric_column])
                
            analysis_df.loc[mask_inf, :] = np.nan
            
            # Drop rows with NaN in critical columns
            mask_nan = analysis_df[signal_column].isna() | analysis_df[metric_column].isna()
            analysis_df = analysis_df[~mask_nan]
            
            # Count dropped rows
            dropped_rows = original_row_count - len(analysis_df)
            if dropped_rows > 0:
                st.warning(f"Dropping {dropped_rows} rows ({dropped_rows/original_row_count:.1%} of data) with NaN/Inf values in '{signal_column}' or '{metric_column}'")
            
            # Check if signal column has numeric values
            is_numeric_signal = pd.api.types.is_numeric_dtype(analysis_df[signal_column])
            
            if is_numeric_signal:
                st.write(f"Signal column '{signal_column}' contains numeric values. Will perform correlation analysis.")
                analysis_type = "correlation"
                signal_values = []  # Not needed for correlation
                analyze_all_values = []
            else:
                st.write(f"Signal column '{signal_column}' contains categorical values. Will compare groups.")
                analysis_type = "group_comparison"
                
                # Add toggle for analyzing all values
                analyze_all = st.checkbox("Analyze all signal values individually", key="analyze_all_checkbox")
                
                if analyze_all:
                    # For "analyze all", we'll run multiple analyses, one for each value
                    try:
                        unique_signals = pd.Series(analysis_df[signal_column].dropna().values).unique()
                        st.write(f"Will analyze each of the {len(unique_signals)} unique values compared to all others")
                        signal_values = [unique_signals[0]]  # Start with first value, we'll iterate through all later
                        analyze_all_values = list(unique_signals)
                    except:
                        st.error(f"Error processing unique values in {signal_column}. Please check column data.")
                        signal_values = []
                        analyze_all_values = []
                else:
                    # Manual selection
                    analyze_all_values = []
                    try:
                        unique_signals = pd.Series(analysis_df[signal_column].dropna().values).unique()
                        signal_values = st.multiselect(f"Select signal value(s) to analyze", unique_signals, key="signal_vals_select")
                    except:
                        st.error(f"Error processing unique values in {signal_column}. Please check column data.")
                        signal_values = []
            
            # Metric value selection (if categorical)
            if analysis_df[metric_column].dtype == 'object':
                st.write("Your metric column contains categorical data. Please select the 'success' value:")
                try:
                    unique_metrics = pd.Series(analysis_df[metric_column].dropna().values).unique()
                    positive_value = st.selectbox("Select the positive outcome value (e.g., 'Yes', 'True', '1')", unique_metrics, key="pos_val_select")
                except:
                    st.error(f"Error processing unique values in {metric_column}. Please check column data.")
                    positive_value = None
                
                if positive_value is not None:
                    # Convert to binary
                    analysis_df['metric_binary'] = (analysis_df[metric_column] == positive_value).astype(int)
                else:
                    analysis_df['metric_binary'] = 0
            else:
                # Numeric column, assume higher is better
                analysis_df['metric_binary'] = analysis_df[metric_column]
            
            # Simple button instead of form
            if st.button("Analyze Data", key="analyze_button"):
                if analysis_type == "group_comparison" and not signal_values and not analyze_all_values:
                    st.error("Please select at least one signal value")
                else:
                    # Replace the existing correlation analysis code with this:
                    if analysis_type == "correlation":
                        # Perform correlation analysis
                        # Ensure metric_binary is numeric
                        if analysis_df['metric_binary'].dtype == 'object':
                            st.error("Cannot perform correlation with non-numeric metric column")
                        else:
                            # Identify outliers
                            signal_outliers = identify_outliers(analysis_df, signal_column)
                            metric_outliers = identify_outliers(analysis_df, 'metric_binary')
                            combined_outliers = signal_outliers.union(metric_outliers)
                            
                            # Calculate Pearson correlation
                            pearson_corr, pearson_p = pearsonr(analysis_df[signal_column], analysis_df['metric_binary'])
                            
                            # Calculate Spearman rank correlation
                            spearman_corr, spearman_p = spearmanr(analysis_df[signal_column], analysis_df['metric_binary'])
                            
                            # Calculate confidence intervals
                            pearson_ci_lower, pearson_ci_upper = correlation_ci(pearson_corr, len(analysis_df))
                            spearman_ci_lower, spearman_ci_upper = correlation_ci(spearman_corr, len(analysis_df))
                            
                            # Create a version without outliers for comparison
                            if len(combined_outliers) > 0:
                                clean_df = analysis_df.drop(combined_outliers)
                                pearson_clean, pearson_p_clean = pearsonr(clean_df[signal_column], clean_df['metric_binary'])
                                spearman_clean, spearman_p_clean = spearmanr(clean_df[signal_column], clean_df['metric_binary'])
                            else:
                                pearson_clean, pearson_p_clean = pearson_corr, pearson_p
                                spearman_clean, spearman_p_clean = spearman_corr, spearman_p
                            
                            # Store results in session state
                            st.session_state.signal_results = {
                                'df': analysis_df,
                                'signal_column': signal_column,
                                'metric_column': metric_column,
                                'analysis_type': 'correlation',
                                'pearson_correlation': pearson_corr,
                                'pearson_p_value': pearson_p,
                                'pearson_ci_lower': pearson_ci_lower,
                                'pearson_ci_upper': pearson_ci_upper,
                                'spearman_correlation': spearman_corr,
                                'spearman_p_value': spearman_p,
                                'spearman_ci_lower': spearman_ci_lower,
                                'spearman_ci_upper': spearman_ci_upper,
                                'outliers_count': len(combined_outliers),
                                'pearson_no_outliers': pearson_clean,
                                'pearson_p_no_outliers': pearson_p_clean,
                                'spearman_no_outliers': spearman_clean,
                                'spearman_p_no_outliers': spearman_p_clean,
                                'dropped_rows': dropped_rows,
                                'original_row_count': original_row_count
                            }
                            st.session_state.has_results = True
                    
                    elif analysis_type == "group_comparison":
                        if analyze_all_values:
                            # We'll create a separate analysis for each value
                            all_results = []
                            
                            for signal_val in analyze_all_values:
                                # Split data for this specific value vs all others
                                signal_specific = analysis_df[analysis_df[signal_column] == signal_val]
                                non_signal_specific = analysis_df[analysis_df[signal_column] != signal_val]
                                
                                # Skip if either group is empty
                                if len(signal_specific) == 0 or len(non_signal_specific) == 0:
                                    continue
                                    
                                # Calculate metrics
                                signal_rate = signal_specific['metric_binary'].mean()
                                non_signal_rate = non_signal_specific['metric_binary'].mean()
                                
                                # Calculate lift
                                if non_signal_rate == 0:
                                    lift_val = 0
                                else:
                                    lift_val = signal_rate / non_signal_rate - 1
                                    
                                # Calculate statistical significance
                                n1_val = len(signal_specific)
                                n2_val = len(non_signal_specific)
                                p1_val = signal_rate
                                p2_val = non_signal_rate
                                
                                # Get p-value
                                if n1_val > 0 and n2_val > 0:
                                    # Pooled proportion
                                    p_pooled_val = (signal_specific['metric_binary'].sum() + non_signal_specific['metric_binary'].sum()) / (n1_val + n2_val)
                                    
                                    if p_pooled_val == 0 or p_pooled_val == 1:
                                        p_value_val = 1.0
                                    else:
                                        se_val = np.sqrt(p_pooled_val * (1 - p_pooled_val) * (1/n1_val + 1/n2_val))
                                        z_val = (p1_val - p2_val) / se_val if se_val > 0 else 0
                                        p_value_val = 2 * (1 - stats.norm.cdf(abs(z_val)))
                                else:
                                    p_value_val = 1.0
                                
                                # Save result
                                all_results.append({
                                    'Signal Value': signal_val,
                                    'Signal Count': n1_val,
                                    'Non-Signal Count': n2_val,
                                    'Signal Rate': signal_rate,
                                    'Non-Signal Rate': non_signal_rate,
                                    'Lift': lift_val,
                                    'P-value': p_value_val,
                                    'Significant': p_value_val < 0.05
                                })
                            
                            # Store results differently for all-values analysis
                            st.session_state.signal_results = {
                                'df': analysis_df,
                                'signal_column': signal_column,
                                'metric_column': metric_column,
                                'analysis_type': 'all_values_comparison',
                                'all_results': all_results,
                                'dropped_rows': dropped_rows,
                                'original_row_count': original_row_count
                            }
                            st.session_state.has_results = True
                        else:
                            # Normal group comparison with selected values
                            # Split data into signal and non-signal groups
                            signal_group = analysis_df[analysis_df[signal_column].isin(signal_values)]
                            non_signal_group = analysis_df[~analysis_df[signal_column].isin(signal_values)]
                            
                            # Calculate metrics
                            signal_conversion = signal_group['metric_binary'].mean()
                            non_signal_conversion = non_signal_group['metric_binary'].mean()
                            overall_conversion = analysis_df['metric_binary'].mean()
                            
                            # Calculate lift
                            if non_signal_conversion == 0:
                                lift = 0  # Handle division by zero
                                st.warning("Non-signal conversion rate is zero. Cannot calculate proper lift.")
                            else:
                                lift = signal_conversion / non_signal_conversion - 1
                            
                            # Calculate statistical significance
                            # Z-test for proportions
                            n1 = len(signal_group)
                            n2 = len(non_signal_group)
                            p1 = signal_conversion
                            p2 = non_signal_conversion
                            
                            # Pooled proportion - Check for empty groups first
                            if n1 == 0 or n2 == 0:
                                st.warning("One of the groups is empty. Cannot calculate statistical significance.")
                                p_pooled = 0
                                se = 0
                                z_score = 0
                                p_value = 1.0  # Not significant by default
                            else:
                                p_pooled = (signal_group['metric_binary'].sum() + non_signal_group['metric_binary'].sum()) / (n1 + n2)
                                
                                # Handle edge cases to prevent division by zero
                                if p_pooled == 0 or p_pooled == 1:
                                    se = 0
                                    z_score = 0
                                    p_value = 1.0  # Not significant by default
                                else:
                                    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                                    z_score = (p1 - p2) / se
                                    # P-value (two-tailed test)
                                    p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                            
                            # Calculate Chi-square test
                            contingency_table = pd.DataFrame({
                                'Success': [signal_group['metric_binary'].sum(), non_signal_group['metric_binary'].sum()],
                                'Failure': [(signal_group['metric_binary'] == 0).sum(), (non_signal_group['metric_binary'] == 0).sum()]
                            }, index=['Signal', 'Non-Signal'])

                            chi2, chi2_p, dof, expected = chi2_contingency(contingency_table)

                            # Store Chi-square results in session state
                            st.session_state.signal_results.update({
                                'chi2_value': chi2,
                                'chi2_p_value': chi2_p,
                                'chi2_dof': dof
                            })
                            
                            # Store results in session state
                            st.session_state.signal_results = {
                                'df': analysis_df,
                                'signal_column': signal_column,
                                'metric_column': metric_column,
                                'signal_values': signal_values,
                                'signal_conversion': signal_conversion,
                                'non_signal_conversion': non_signal_conversion,
                                'overall_conversion': overall_conversion,
                                'lift': lift,
                                'z_score': z_score,
                                'p_value': p_value,
                                'n1': n1,
                                'n2': n2,
                                'analysis_type': 'group_comparison',
                                'dropped_rows': dropped_rows,
                                'original_row_count': original_row_count
                            }
                            st.session_state.has_results = True
            
        except Exception as e:
            st.error(f"Error processing file: {e}")

        # Show results if analysis was performed
        if st.session_state.has_results:
            results = st.session_state.signal_results
            df = results['df']
            
            # Display results
            st.header("Analysis Results")
            
            # Allow user to set significance level
            significance_level = st.slider(
                "Select significance level (p-value threshold)",
                min_value=0.05,
                max_value=0.20,
                value=0.05,
                step=0.01,
                format="%.2f",
                key="significance_slider"
            )
            alpha = significance_level
            confidence_level = (1 - alpha) * 100
            
            # Use the original dataframe (st.session_state.df) for a chi-square test on unaltered categorical data.
            if st.session_state.df[signal_column].dtype == 'object' and st.session_state.df[metric_column].dtype == 'object':
                st.header("Chi-Square Test for Independence")
                if st.checkbox("Perform Chi-Square Test", key="chi_square_checkbox", value=True):
                    contingency_table, chi2, p, dof, expected = run_chi_square_test(
                        st.session_state.df[signal_column].dropna(),
                        st.session_state.df[metric_column].dropna()
                    )
                    st.write("### Contingency Table")
                    st.dataframe(contingency_table)
                    st.write(f"**Chi-Square Statistic:** {chi2:.4f}")
                    st.write(f"**Degrees of Freedom:** {dof}")
                    st.write(f"**p-value:** {p:.4f}")
                    
                    st.subheader("Contingency Table Heatmap")
                    plot_chi_square_heatmap(contingency_table)
                    # Convert the contingency table to a dictionary
                    data_dict = contingency_table.stack().to_dict()

                    fig, ax = plt.subplots(figsize=(10, 8))
                    mosaic(data_dict, title='Mosaic Plot of Signal vs. Metric', ax=ax)
                    plt.tight_layout()

                    st.pyplot(fig)

            st.write(f"Using {confidence_level:.0f}% confidence level (p < {alpha:.2f})")
            if results.get('analysis_type') == 'correlation':
                    # Display correlation results
                    st.subheader(f"Correlation Analysis: {results['signal_column']} vs {results['metric_column']}")
                    
                    # Pearson correlation results
                    st.write("### Linear (Pearson) Correlation")
                    
                    pearson_corr = results.get('pearson_correlation', results.get('correlation', 0))
                    pearson_p = results.get('pearson_p_value', results.get('p_value', 1))
                    
                    col1, col2 = st.columns(2)
                    with col1:
                        st.metric("Pearson Correlation", f"{pearson_corr:.4f}")
                        if 'pearson_ci_lower' in results and 'pearson_ci_upper' in results:
                            st.write(f"95% CI: [{results['pearson_ci_lower']:.4f}, {results['pearson_ci_upper']:.4f}]")
                    with col2:
                        st.metric("P-value", f"{pearson_p:.4f}")
                    
                    # Interpretation
                    if pearson_p < alpha:
                        significance = "statistically significant"
                        st.success(f"The linear correlation is {significance} (p < {alpha:.2f}).")
                    else:
                        significance = "NOT statistically significant"
                        st.warning(f"The linear correlation is {significance} (p > {alpha:.2f}).")
                    
                    # Correlation strength interpretation
                    abs_corr = abs(pearson_corr)
                    if abs_corr < 0.1:
                        strength = "negligible"
                    elif abs_corr < 0.3:
                        strength = "weak"
                    elif abs_corr < 0.5:
                        strength = "moderate"
                    elif abs_corr < 0.7:
                        strength = "strong"
                    else:
                        strength = "very strong"
                    
                    direction = "positive" if pearson_corr > 0 else "negative"
                    
                    if strength == "negible":
                        st.write(f"The Pearson correlation is {strength} but {direction} ({pearson_corr:.4f}).")
                    else:
                        st.write(f"The Pearson correlation is {strength} and {direction} ({pearson_corr:.4f}).")
                    
                    # Spearman correlation results (if available)
                    if 'spearman_correlation' in results:
                        st.write("### Monotonic (Spearman) Correlation")
                        
                        spearman_corr = results['spearman_correlation']
                        spearman_p = results['spearman_p_value']
                        
                        col1, col2 = st.columns(2)
                        with col1:
                            st.metric("Spearman Rank Correlation", f"{spearman_corr:.4f}")
                            if 'spearman_ci_lower' in results and 'spearman_ci_upper' in results:
                                st.write(f"95% CI: [{results['spearman_ci_lower']:.4f}, {results['spearman_ci_upper']:.4f}]")
                        with col2:
                            st.metric("P-value", f"{spearman_p:.4f}")
                        
                        # Interpretation
                        if spearman_p < alpha:
                            significance = "statistically significant"
                            st.success(f"The monotonic correlation is {significance} (p < {alpha:.2f}).")
                        else:
                            significance = "NOT statistically significant"
                            st.warning(f"The monotonic correlation is {significance} (p > {alpha:.2f}).")
                        
                        # Correlation strength interpretation
                        abs_corr = abs(spearman_corr)
                        if abs_corr < 0.1:
                            strength = "negligible"
                        elif abs_corr < 0.3:
                            strength = "weak"
                        elif abs_corr < 0.5:
                            strength = "moderate"
                        elif abs_corr < 0.7:
                            strength = "strong"
                        else:
                            strength = "very strong"
                        
                        direction = "positive" if spearman_corr > 0 else "negative"
                        
                        st.write(f"The Spearman correlation is {strength} and {direction} ({spearman_corr:.4f}).")
                        
                        # Compare Pearson and Spearman
                        pearson_diff = abs(pearson_corr) - abs(spearman_corr)
                        if abs(pearson_diff) > 0.1:
                            if pearson_diff < 0:
                                st.info(f"The Spearman correlation is stronger than Pearson by {abs(pearson_diff):.2f}, suggesting the relationship may be monotonic but non-linear.")
                            else:
                                st.info(f"The Pearson correlation is stronger than Spearman by {abs(pearson_diff):.2f}, suggesting the relationship is primarily linear.")
                    
                    # Outlier analysis (if available)
                    if 'outliers_count' in results:
                        st.write("### Outlier Analysis")
                        
                        outliers_count = results['outliers_count']
                        st.write(f"Identified {outliers_count} potential outliers ({outliers_count/len(results['df']):.1%} of data)")
                        
                        if outliers_count > 0 and 'pearson_no_outliers' in results:
                            col1, col2 = st.columns(2)
                            
                            with col1:
                                st.write("**With Outliers**")
                                st.write(f"Pearson: {pearson_corr:.4f} (p={pearson_p:.4f})")
                                if 'spearman_correlation' in results:
                                    st.write(f"Spearman: {spearman_corr:.4f} (p={spearman_p:.4f})")
                                
                            with col2:
                                st.write("**Without Outliers**")
                                st.write(f"Pearson: {results['pearson_no_outliers']:.4f} (p={results['pearson_p_no_outliers']:.4f})")
                                if 'spearman_no_outliers' in results:
                                    st.write(f"Spearman: {results['spearman_no_outliers']:.4f} (p={results['spearman_p_no_outliers']:.4f})")
                            
                            # Impact assessment
                            pearson_impact = abs(pearson_corr - results['pearson_no_outliers'])
                            if 'spearman_correlation' in results and 'spearman_no_outliers' in results:
                                spearman_impact = abs(spearman_corr - results['spearman_no_outliers'])
                                impact = max(pearson_impact, spearman_impact)
                            else:
                                impact = pearson_impact
                            
                            if impact > 0.1:
                                st.warning(f"Outliers substantially impact the correlation results (Δ{impact:.2f}). Consider handling them carefully.")
                            else:
                                st.success(f"Outliers have minimal impact on the correlation results.")
                    
                    # Visualization - Scatter plot (kept from original code)
                    st.subheader("Scatter Plot")
                    
                    # Create scatter plot data
                    scatter_data = pd.DataFrame({
                        results['signal_column']: df[results['signal_column']],
                        results['metric_column']: df['metric_binary']
                    })
                    
                    chart = st.scatter_chart(
                        scatter_data,
                        x=results['signal_column'],
                        y=results['metric_column']
                    )
            
            elif results.get('analysis_type') == 'all_values_comparison':
                # All values comparison results (new analysis type)
                st.subheader("Results for Each Signal Value")
                
                # Get and sort results
                all_results = results['all_results']
                sorted_results = sorted(all_results, key=lambda x: x['Lift'], reverse=True)
                
                # Update significance based on current alpha
                for result in sorted_results:
                    result['Significant'] = result['P-value'] < alpha
                
                # Format for display
                display_df = pd.DataFrame(sorted_results)
                display_df['Signal Rate'] = display_df['Signal Rate'].apply(lambda x: f"{x:.2%}")
                display_df['Non-Signal Rate'] = display_df['Non-Signal Rate'].apply(lambda x: f"{x:.2%}")
                display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.2%}")
                display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f}")
                display_df['Significant'] = display_df['Significant'].apply(lambda x: "✓" if x else "")
                
                st.dataframe(display_df)
                
                # Create chart for top values by lift
                top_n = min(10, len(sorted_results))
                top_values = pd.DataFrame(sorted_results[:top_n])
                
                if not top_values.empty:
                    st.subheader(f"Top {top_n} Signal Values by Lift")
                    
                    chart_data = pd.DataFrame({
                        'Signal Value': top_values['Signal Value'].astype(str),
                        'Lift (%)': top_values['Lift'] * 100  # Convert to percentage points
                    })
                    
                    st.bar_chart(chart_data.set_index('Signal Value'))
            
            else:
                # Group comparison results display (existing code)
                col1, col2, col3 = st.columns(3)
                with col1:
                    signal_label = ", ".join(str(v) for v in results['signal_values']) if len(results['signal_values']) <= 3 else f"{len(results['signal_values'])} selected values"
                    st.metric(f"Selected Signal(s) Conversion Rate", f"{results['signal_conversion']:.2%}")
                with col2:
                    st.metric("Non-Signal Conversion Rate", f"{results['non_signal_conversion']:.2%}")
                with col3:
                    st.metric("Overall Conversion Rate", f"{results['overall_conversion']:.2%}")
                
                st.metric("Lift", f"{results['lift']:.2%}")
                
                # Interpretation
                st.subheader("Statistical Significance")
                st.write(f"Z-score: {results['z_score']:.2f}")
                st.write(f"P-value: {results['p_value']:.4f}")
                
                if results['p_value'] < alpha:
                    st.success(f"The difference is statistically significant (p < {alpha:.2f}).")
                else:
                    st.warning(f"The difference is NOT statistically significant (p > {alpha:.2f}).")
                
                # Sample size display
                st.subheader("Sample Sizes")
                st.write(f"Selected Signal(s) group: {results['n1']} samples")
                st.write(f"Non-Signal group: {results['n2']} samples")
                
                # Show selected values
                st.subheader("Selected Signal Values")
                st.write(", ".join(str(v) for v in results['signal_values']))
                
                # Visualization
                st.subheader("Overall Comparison")
                
                # Create a bar chart
                signal_label = ", ".join(str(v) for v in results['signal_values']) if len(results['signal_values']) <= 3 else f"{len(results['signal_values'])} selected values"
                data = pd.DataFrame({
                    'Group': [f"Selected Signal(s)", "Non-Signal"],
                    'Conversion Rate': [results['signal_conversion'], results['non_signal_conversion']]
                })
                
                chart = st.bar_chart(data.set_index('Group'))
            
            # Breakdown by category
            st.header("Breakdown Analysis")
            st.write("Filter results by a specific column to see how the signal performs across different categories")
            
            # Get available columns for breakdown (excluding signal and metric columns)
            breakdown_columns = [col for col in df.columns.tolist() 
                            if col != results['signal_column'] 
                            and col != results['metric_column']
                            and col != 'metric_binary'
                            and len(df[col].unique()) < 50]  # Limit to columns with fewer unique values
            
            if breakdown_columns:
                # Two-column layout for breakdown controls
                col1, col2 = st.columns([3, 1])
                
                with col1:
                    breakdown_column = st.selectbox("Select column to filter by:", 
                                                breakdown_columns,
                                                key="breakdown_column_selector")
                
                with col2:
                    st.write(" ")  # Spacing
                    st.write(" ")  # Spacing
                    generate_breakdown = st.button("Show Breakdown", key="generate_breakdown_button")
                
                if generate_breakdown and breakdown_column:
                    st.subheader(f"Breakdown by {breakdown_column}")
                    
                    # Get unique values in breakdown column
                    try:
                        breakdown_values = sorted(pd.Series(df[breakdown_column].dropna().values).unique())
                    except:
                        st.warning(f"Error processing unique values in {breakdown_column}. Using alternative method.")
                        breakdown_values = sorted(df[breakdown_column].dropna().unique())
                    
                    # Create a DataFrame to store breakdown results
                    breakdown_results = []
                    
                    # Calculate metrics for each breakdown value
                    skipped_values = []
                    total_values = len(breakdown_values)
                    
                    for value in breakdown_values:
                        # Filter data for this breakdown value
                        value_df = df[df[breakdown_column] == value]
                        
                        # Skip if too few samples
                        if len(value_df) < 5:  # Lowered threshold to show more results
                            skipped_values.append(f"{value} (too few samples: {len(value_df)})")
                            continue
                        
                        if results.get('analysis_type') == 'correlation':
                            # Skip if not enough data points for correlation
                            if len(value_df) < 10:
                                skipped_values.append(f"{value} (insufficient data for correlation: {len(value_df)})")
                                continue
                                
                            # Calculate correlation for this value
                            try:
                                from scipy.stats import pearsonr
                                value_corr, value_p = pearsonr(value_df[results['signal_column']], value_df['metric_binary'])
                                
                                # Add to results
                                breakdown_results.append({
                                    'Category': value,
                                    'Count': len(value_df),
                                    'Correlation': value_corr,
                                    'P-value': value_p,
                                    'Significant': value_p < alpha
                                })
                            except Exception as e:
                                skipped_values.append(f"{value} (error: {str(e)})")
                                continue
                        elif results.get('analysis_type') == 'all_values_comparison':
                            # For all values comparison breakdown
                            value_results = []
                            
                            for signal_val in sorted_results:
                                signal_value = signal_val['Signal Value']
                                
                                # Filter to just this category and signal value
                                value_signal = value_df[value_df[results['signal_column']] == signal_value]
                                value_non_signal = value_df[value_df[results['signal_column']] != signal_value]
                                
                                # Skip if any group is empty
                                if len(value_signal) == 0 or len(value_non_signal) == 0:
                                    continue
                                    
                                # Calculate metrics
                                value_signal_rate = value_signal['metric_binary'].mean()
                                value_non_signal_rate = value_non_signal['metric_binary'].mean()
                                
                                # Handle division by zero
                                if value_non_signal_rate == 0:
                                    value_lift = 0
                                else:
                                    value_lift = value_signal_rate / value_non_signal_rate - 1
                                    
                                # Calculate p-value
                                n1 = len(value_signal)
                                n2 = len(value_non_signal)
                                
                                if n1 < 5 or n2 < 5:
                                    continue  # Skip if either group is too small
                                    
                                # Pooled proportion
                                p_pooled = (value_signal['metric_binary'].sum() + value_non_signal['metric_binary'].sum()) / (n1 + n2)
                                
                                # Handle edge cases
                                if p_pooled == 0 or p_pooled == 1:
                                    value_p = 1.0
                                    is_significant = False
                                else:
                                    # Standard error
                                    se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                                    
                                    # Z-score
                                    z = (value_signal_rate - value_non_signal_rate) / se if se > 0 else 0
                                    
                                    # P-value (two-tailed test)
                                    value_p = 2 * (1 - stats.norm.cdf(abs(z)))
                                    is_significant = value_p < alpha
                                
                                if 'chi2_value' in results:
                                    st.subheader("Chi-square Test")
                                    st.write(f"Chi-square value: {results['chi2_value']:.2f}")
                                    st.write(f"Degrees of freedom: {results['chi2_dof']}")
                                    st.write(f"P-value: {results['chi2_p_value']:.4f}")
                                    
                                    if results['chi2_p_value'] < alpha:
                                        st.success(f"The Chi-square test shows a statistically significant association (p < {alpha:.2f}).")
                                    else:
                                        st.warning(f"The Chi-square test does NOT show a statistically significant association (p > {alpha:.2f}).")
                                    
                                    # Explain the difference between Z-test and Chi-square test results if they differ
                                    if (results['p_value'] < alpha) != (results['chi2_p_value'] < alpha):
                                        st.info("Note: The Z-test and Chi-square test gave different significance results. This can happen due to different test assumptions. The Z-test specifically tests the difference in proportions, while the Chi-square test examines the association between variables.")

                                # Add to results
                                value_results.append({
                                    'Category': value,
                                    'Signal Value': signal_value,
                                    'Signal Count': n1,
                                    'Non-Signal Count': n2,
                                    'Signal Rate': value_signal_rate,
                                    'Non-Signal Rate': value_non_signal_rate,
                                    'Lift': value_lift,
                                    'P-value': value_p,
                                    'Significant': is_significant
                                })
                                
                            # Add all value results to the main results
                            breakdown_results.extend(value_results)
                        else:
                            # Calculate metrics for group comparison
                            value_signal = value_df[value_df[results['signal_column']].isin(results['signal_values'])]
                            value_non_signal = value_df[~value_df[results['signal_column']].isin(results['signal_values'])]
                            
                            # Skip if any group is empty
                            if len(value_signal) == 0 or len(value_non_signal) == 0:
                                if len(value_signal) == 0:
                                    skipped_values.append(f"{value} (no signal group samples)")
                                else:
                                    skipped_values.append(f"{value} (no non-signal group samples)")
                                continue
                            
                            value_signal_rate = value_signal['metric_binary'].mean()
                            value_non_signal_rate = value_non_signal['metric_binary'].mean()
                            
                            # Handle division by zero
                            if value_non_signal_rate == 0:
                                value_lift = 0
                            else:
                                value_lift = value_signal_rate / value_non_signal_rate - 1
                            
                            # Calculate p-value
                            n1 = len(value_signal)
                            n2 = len(value_non_signal)
                            p1 = value_signal_rate
                            p2 = value_non_signal_rate
                            
                            # Pooled proportion
                            p_pooled = (value_signal['metric_binary'].sum() + value_non_signal['metric_binary'].sum()) / (n1 + n2)
                            
                            # Handle edge cases
                            if p_pooled == 0 or p_pooled == 1 or n1 == 0 or n2 == 0:
                                value_p = 1.0
                                is_significant = False
                            else:
                                # Standard error
                                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                                
                                # Z-score
                                z = (p1 - p2) / se if se > 0 else 0
                                
                                # P-value (two-tailed test)
                                value_p = 2 * (1 - stats.norm.cdf(abs(z)))
                                is_significant = value_p < alpha
                            
                            # Add to results
                            breakdown_results.append({
                                'Category': value,
                                'Signal Count': n1,
                                'Non-Signal Count': n2,
                                'Signal Rate': value_signal_rate,
                                'Non-Signal Rate': value_non_signal_rate,
                                'Lift': value_lift,
                                'P-value': value_p,
                                'Significant': is_significant
                            })
                    
                    # Convert to DataFrame and display
                    if breakdown_results:
                        # Extract p-values from breakdown results
                        p_values = [result['P-value'] for result in breakdown_results]
                        
                        # Apply multiple testing correction (Benjamini-Hochberg FDR)
                        reject, p_corrected, _, _ = multipletests(p_values, alpha=alpha, method='fdr_bh')
                        
                        # Update results with corrected p-values and significance
                        for i, result in enumerate(breakdown_results):
                            result['Original_P'] = result['P-value']  # Store original p-value
                            result['P-value'] = p_corrected[i]  # Update with corrected p-value
                            result['Significant'] = reject[i]  # Update significance based on correction
                        # Store results in session state for persistence
                        st.session_state.breakdown_results = breakdown_results
                        
                        # Report on skipped values
                        if skipped_values:
                            st.info(f"Skipped {len(skipped_values)} of {total_values} categories due to data issues:")
                            with st.expander("Show skipped categories"):
                                for skipped in skipped_values:
                                    st.write(f"- {skipped}")
                        
                        if results.get('analysis_type') == 'all_values_comparison':
                            # Special format for all values comparison - need to group by category
                            display_df = pd.DataFrame(breakdown_results)
                            
                            # Convert Signal Value to string in the dataframe itself
                            display_df['Signal Value'] = display_df['Signal Value'].astype(str)
                            
                            # Get unique values and sort
                            signal_values_in_breakdown = sorted(display_df['Signal Value'].unique())
                            
                            # Use session state to keep selected value persistent
                            if st.session_state.selected_breakdown_signal not in signal_values_in_breakdown:
                                st.session_state.selected_breakdown_signal = signal_values_in_breakdown[0] if signal_values_in_breakdown else None
                            
                            # Create the selectbox with the current value
                            selected_signal = st.selectbox(
                                "Filter by signal value:", 
                                options=signal_values_in_breakdown,
                                index=signal_values_in_breakdown.index(st.session_state.selected_breakdown_signal) if st.session_state.selected_breakdown_signal in signal_values_in_breakdown else 0,
                                key="breakdown_signal_filter",
                                on_change=lambda: setattr(st.session_state, 'selected_breakdown_signal', st.session_state.breakdown_signal_filter)
                            )
                            
                            # Filter using the string version
                            filtered_df = display_df[display_df['Signal Value'] == selected_signal].copy()
                            
                        

                            
                            # Format for display
                            filtered_df['Signal Rate'] = filtered_df['Signal Rate'].apply(lambda x: f"{x:.2%}")
                            filtered_df['Non-Signal Rate'] = filtered_df['Non-Signal Rate'].apply(lambda x: f"{x:.2%}")
                            filtered_df['Lift'] = filtered_df['Lift'].apply(lambda x: f"{x:.2%}")
                            filtered_df['P-value'] = filtered_df['P-value'].apply(lambda x: f"{x:.4f}")
                            filtered_df['Significant'] = filtered_df['Significant'].apply(lambda x: "✓" if x else "")
                            
                            # Show the filtered dataframe
                            st.dataframe(filtered_df.sort_values('Lift', ascending=False))
                            
                            # Create chart for top categories by lift for this signal value
                            sorted_df = filtered_df.sort_values('Lift', ascending=False)
                            raw_df = display_df[display_df['Signal Value'] == selected_signal].sort_values('Lift', ascending=False)
                            
                            top_n = min(10, len(raw_df))
                            if top_n > 0:
                                st.subheader(f"Top Categories for Signal Value: {selected_signal}")
                                
                                chart_data = pd.DataFrame({
                                    'Category': raw_df['Category'].astype(str).head(top_n),
                                    'Lift (%)': raw_df['Lift'].head(top_n) * 100
                                })
                                
                                st.bar_chart(chart_data.set_index('Category'))
                        
                        elif results.get('analysis_type') == 'correlation':
                            # Format for correlation display
                            results_df = pd.DataFrame(breakdown_results)
                            display_df = results_df.copy()
                            display_df['Correlation'] = display_df['Correlation'].apply(lambda x: f"{x:.4f}")
                            display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f}")
                            display_df['Significant'] = display_df['Significant'].apply(lambda x: "✓" if x else "")
                            
                            st.dataframe(display_df.sort_values('Correlation', ascending=False))
                            
                            # Create chart for top categories by correlation
                            top_n = min(10, len(results_df))
                            top_categories = results_df.sort_values('Correlation', ascending=False).head(top_n)
                            
                            if not top_categories.empty:
                                st.subheader(f"Top Categories by Correlation in {breakdown_column}")
                                
                                chart_data = pd.DataFrame({
                                    'Category': top_categories['Category'].astype(str),
                                    'Correlation': top_categories['Correlation']
                                })
                                
                                chart = st.bar_chart(chart_data.set_index('Category'))
                        else:
                            # Format for group comparison display
                            results_df = pd.DataFrame(breakdown_results)
                            display_df = results_df.copy()
                            display_df['Signal Rate'] = display_df['Signal Rate'].apply(lambda x: f"{x:.2%}")
                            display_df['Non-Signal Rate'] = display_df['Non-Signal Rate'].apply(lambda x: f"{x:.2%}")
                            display_df['Lift'] = display_df['Lift'].apply(lambda x: f"{x:.2%}")
                            display_df['P-value'] = display_df['P-value'].apply(lambda x: f"{x:.4f}")
                            display_df['Significant'] = display_df['Significant'].apply(lambda x: "✓" if x else "")
                            
                            st.dataframe(display_df.sort_values('Lift', ascending=False))
                            
                            # Create chart for top categories by lift
                            top_n = min(10, len(results_df))
                            top_categories = results_df.sort_values('Lift', ascending=False).head(top_n)
                            
                            if not top_categories.empty:
                                st.subheader(f"Top Categories by Lift in {breakdown_column}")
                                
                                chart_data = pd.DataFrame({
                                    'Category': top_categories['Category'].astype(str),
                                    'Lift (%)': top_categories['Lift'] * 100  # Convert to percentage points
                                })
                                
                                chart = st.bar_chart(chart_data.set_index('Category'))
                    else:
                        st.warning(f"Not enough data to show breakdown by {breakdown_column}")
            else:
                st.info("No suitable columns found for breakdown analysis.")

    if st.session_state.has_results:
        results = st.session_state.signal_results
        
        
        
        # Check if results are not statistically significant
        is_significant = False
        
        if results.get('analysis_type') == 'correlation':
            is_significant = results.get('pearson_p_value', 1) < alpha
        elif results.get('analysis_type') == 'all_values_comparison':
            # For all values comparison, check if any value is significant
            all_results = results.get('all_results', [])
            is_significant = any(result.get('P-value', 1) < alpha for result in all_results)
        else:
            # Group comparison
            is_significant = results.get('p_value', 1) < alpha
        
        # If results are not statistically significant, show sample size calculator
        if not is_significant:
            st.header("Sample Size Calculator")
            st.info("Your analysis doesn't show statistical significance. Let's calculate how many more samples you might need.")
            
            # Calculate the current effect size
            if results.get('analysis_type') == 'correlation':
                # For correlation, use the observed correlation as effect size
                observed_effect = abs(results.get('pearson_correlation', 0))
                baseline_rate = results['df']['metric_binary'].mean()
                
                st.write(f"Current correlation: {observed_effect:.4f}")
                st.write(f"Current baseline rate: {baseline_rate:.2%}")
                
                # UI for sample size calculation
                min_effect = st.slider(
                    "Minimum effect size to detect",
                    min_value=max(0.01, observed_effect/2),
                    max_value=max(0.5, observed_effect*2),
                    value=observed_effect,
                    step=0.01,
                    format="%.2f",
                    help="The correlation strength you want to detect"
                )
                
            elif results.get('analysis_type') == 'all_values_comparison':
                # Find the largest effect size in the current results
                lift_values = [abs(result.get('Lift', 0)) for result in results.get('all_results', [])]
                observed_effect = max(lift_values) if lift_values else 0.1
                baseline_rate = results['df']['metric_binary'].mean()
                
                st.write(f"Current maximum lift: {observed_effect:.2%}")
                st.write(f"Current baseline rate: {baseline_rate:.2%}")
                
                # UI for sample size calculation
                min_effect = st.slider(
                    "Minimum lift to detect",
                    min_value=max(0.05, observed_effect/2),
                    max_value=max(0.5, observed_effect*2),
                    value=observed_effect,
                    step=0.05,
                    format="%.2f",
                    help="The lift percentage you want to detect"
                )
                
            else:
                # Group comparison
                observed_effect = abs(results.get('lift', 0))
                baseline_rate = results.get('non_signal_conversion', 0.1)
                
                st.write(f"Current observed lift: {observed_effect:.2%}")
                st.write(f"Current baseline rate: {baseline_rate:.2%}")
                
                # UI for sample size calculation
                min_effect = st.slider(
                    "Minimum lift to detect",
                    min_value=max(0.05, observed_effect/2),
                    max_value=max(0.5, observed_effect*2),
                    value=observed_effect,
                    step=0.05,
                    format="%.2f",
                    help="The lift percentage you want to detect"
                )
            
            # Common UI elements for all analysis types
            col1, col2 = st.columns(2)
            
            with col1:
                alpha_sample = st.select_slider(
                    "Significance level (α)",
                    options=[0.01, 0.05, 0.10],
                    value=alpha,
                    help="Probability of false positive (Type I error)"
                )
            
            with col2:
                power = st.select_slider(
                    "Statistical power",
                    options=[0.8, 0.9, 0.95],
                    value=0.8,
                    help="Probability of detecting the effect if it exists (1 - Type II error)"
                )
            
            # Add the sample size calculation function
            def calculate_required_sample_size(baseline_rate, minimum_detectable_effect, alpha=0.05, power=0.8):
                """
                Calculate the sample size needed to detect a minimum lift/drop with given statistical power.
                """
                # Ensure we have valid rates
                baseline_rate = max(0.001, min(0.999, baseline_rate))
                
                # For correlation studies, convert correlation coefficient to an equivalent effect size
                if results.get('analysis_type') == 'correlation':
                    # Using approximate formula based on Cohen's d
                    # For correlation studies we need a different approach
                    z_alpha = stats.norm.ppf(1 - alpha/2)
                    z_power = stats.norm.ppf(power)
                    
                    # Formula for required sample size to detect correlation
                    sample_size = ((z_alpha + z_power) / np.arctanh(minimum_detectable_effect))**2 + 3
                    return int(np.ceil(sample_size))
                
                # For proportion comparisons (group comparison or all values)
                variation_rate = baseline_rate * (1 + minimum_detectable_effect)
                variation_rate = max(0.001, min(0.999, variation_rate))
                
                # Z-scores for alpha and power
                z_alpha = stats.norm.ppf(1 - alpha/2)
                z_power = stats.norm.ppf(power)
                
                # Pooled proportion
                p_pooled = (baseline_rate + variation_rate) / 2
                
                # Calculate sample size per group
                sample_size = (
                    ((z_alpha + z_power)**2) * 
                    (baseline_rate * (1 - baseline_rate) + variation_rate * (1 - variation_rate)) / 
                    (baseline_rate - variation_rate)**2
                )
                
                return int(np.ceil(sample_size))
            
            # Button to trigger calculation
            if st.button("Calculate Required Sample Size", key="sample_size_btn"):
                total_current_samples = len(results['df'])
                
                required_samples = calculate_required_sample_size(
                    baseline_rate=baseline_rate,
                    minimum_detectable_effect=min_effect,
                    alpha=alpha_sample,
                    power=power
                )
                
                if results.get('analysis_type') != 'correlation':
                    # For group comparisons, we need samples in each group
                    required_samples_total = required_samples * 2
                    
                    # Calculate additional samples needed
                    additional_samples = max(0, required_samples_total - total_current_samples)
                    
                    # Display results
                    st.success(f"You need approximately **{required_samples:,}** samples in each group.")
                    st.write(f"Total samples required: **{required_samples_total:,}**")
                    
                    if additional_samples > 0:
                        st.warning(f"You need **{additional_samples:,}** more samples in total to reach the required sample size.")
                    else:
                        st.success("You already have enough samples! Try adjusting your analysis parameters.")
                else:
                    # For correlation studies, total sample size is what matters
                    additional_samples = max(0, required_samples - total_current_samples)
                    
                    # Display results
                    st.success(f"You need approximately **{required_samples:,}** total samples.")
                    
                    if additional_samples > 0:
                        st.warning(f"You need **{additional_samples:,}** more samples to reach the required sample size.")
                    else:
                        st.success("You already have enough samples! Try adjusting your analysis parameters.")
                
                # Explanation
                st.info(f"""
                **What this means:**
                - With {required_samples:,} samples, you'd have a {power:.0%} chance of detecting a true effect of {min_effect:.2%} or larger.
                - The significance level (α) of {alpha_sample:.0%} means there's a {alpha_sample:.0%} chance of a false positive.
                - Your current sample size is {total_current_samples:,}.
                """)

with tab2:
    st.title("Comparative Analysis")
    st.subheader("Compare signals between control and variation datasets")
    st.write("Upload control and variation files to analyze differences in signal impact on business metrics.")
    
    # Two-column layout for file uploads
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("Control Group")
        control_file = st.file_uploader("Upload control file", type=["csv", "xlsx", "xls"], key="control_file")
        
        if control_file is not None:
            # Load control data
            file_extension = control_file.name.split('.')[-1].lower()
            
            try:
                if file_extension in ['xlsx', 'xls']:
                    control_df = pd.read_excel(control_file)
                else:  # csv
                    control_df = pd.read_csv(control_file)
                
                st.session_state.control_df = control_df
                
                # Display preview
                st.write("Control Data Preview:")
                st.dataframe(control_df.head())
                
                # Show column metadata
                st.write(f"Columns: {control_df.columns.tolist()}")
                st.write(f"Rows: {len(control_df)}")
            except Exception as e:
                st.error(f"Error processing control file: {e}")
    
    with col2:
        st.subheader("Variation Group")
        variation_file = st.file_uploader("Upload variation file", type=["csv", "xlsx", "xls"], key="variation_file")
        
        if variation_file is not None:
            # Load variation data
            file_extension = variation_file.name.split('.')[-1].lower()
            
            try:
                if file_extension in ['xlsx', 'xls']:
                    variation_df = pd.read_excel(variation_file)
                else:  # csv
                    variation_df = pd.read_csv(variation_file)
                
                st.session_state.variation_df = variation_df
                
                # Display preview
                st.write("Variation Data Preview:")
                st.dataframe(variation_df.head())
                
                # Show column metadata
                st.write(f"Columns: {variation_df.columns.tolist()}")
                st.write(f"Rows: {len(variation_df)}")
            except Exception as e:
                st.error(f"Error processing variation file: {e}")
    
    # Only display analysis settings when both datasets are loaded
    if st.session_state.control_df is not None and st.session_state.variation_df is not None:
        st.header("Analysis Settings")
        
        control_df = st.session_state.control_df
        variation_df = st.session_state.variation_df
        
        # Create two columns for control and variation settings
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("Control Settings")
            control_columns = control_df.columns.tolist()
            
            # Select signal column for control
            control_signal_col = st.selectbox(
                "Select signal column (e.g., 'Complexity Score')",
                options=control_columns,
                key="control_signal_select"
            )
            
            # Check if signal column is numeric or categorical
            is_control_signal_numeric = pd.api.types.is_numeric_dtype(control_df[control_signal_col])
            
            if not is_control_signal_numeric:
                st.write(f"Signal column '{control_signal_col}' contains categorical values.")
                unique_control_signals = control_df[control_signal_col].dropna().unique()
                control_signal_value = st.selectbox(
                    "Select signal value to analyze",
                    options=unique_control_signals,
                    key="control_signal_value_select"
                )
            
            # Select metric column for control
            control_metric_col = st.selectbox(
                "Select business metric column (e.g., 'Conversion')",
                options=control_columns,
                key="control_metric_select"
            )
            
            # If metric is categorical, select positive value
            if control_df[control_metric_col].dtype == 'object':
                unique_control_metrics = control_df[control_metric_col].dropna().unique()
                control_positive_value = st.selectbox(
                    "Select the positive outcome value",
                    options=unique_control_metrics,
                    key="control_positive_select"
                )
        
        with col2:
            st.subheader("Variation Settings")
            variation_columns = variation_df.columns.tolist()
            
            # Select signal column for variation
            variation_signal_col = st.selectbox(
                "Select signal column (e.g., 'Complexity Score')",
                options=variation_columns,
                key="variation_signal_select"
            )
            
            # Check if signal column is numeric or categorical
            is_variation_signal_numeric = pd.api.types.is_numeric_dtype(variation_df[variation_signal_col])
            
            if not is_variation_signal_numeric:
                st.write(f"Signal column '{variation_signal_col}' contains categorical values.")
                unique_variation_signals = variation_df[variation_signal_col].dropna().unique()
                variation_signal_value = st.selectbox(
                    "Select signal value to analyze",
                    options=unique_variation_signals,
                    key="variation_signal_value_select"
                )
            
            # Select metric column for variation
            variation_metric_col = st.selectbox(
                "Select business metric column (e.g., 'Conversion')",
                options=variation_columns,
                key="variation_metric_select"
            )
            
            # If metric is categorical, select positive value
            if variation_df[variation_metric_col].dtype == 'object':
                unique_variation_metrics = variation_df[variation_metric_col].dropna().unique()
                variation_positive_value = st.selectbox(
                    "Select the positive outcome value",
                    options=unique_variation_metrics,
                    key="variation_positive_select"
                )
        
        # Analyze button
        if st.button("Run Comparison Analysis", key="run_comparison_button"):
            # Prepare data for analysis
            
            # Handle control data preprocessing
            control_analysis_df = control_df.copy()
            
            # Convert control metric to binary if categorical
            if control_df[control_metric_col].dtype == 'object':
                control_analysis_df['metric_binary'] = (control_analysis_df[control_metric_col] == control_positive_value).astype(int)
            else:
                control_analysis_df['metric_binary'] = control_analysis_df[control_metric_col]
            
            # Handle variation data preprocessing
            variation_analysis_df = variation_df.copy()
            
            # Convert variation metric to binary if categorical
            if variation_df[variation_metric_col].dtype == 'object':
                variation_analysis_df['metric_binary'] = (variation_analysis_df[variation_metric_col] == variation_positive_value).astype(int)
            else:
                variation_analysis_df['metric_binary'] = variation_analysis_df[variation_metric_col]
            
            # For categorical signals, create binary flags
            if not is_control_signal_numeric:
                control_analysis_df['signal_binary'] = (control_analysis_df[control_signal_col] == control_signal_value).astype(int)
            else:
                control_analysis_df['signal_binary'] = control_analysis_df[control_signal_col]
                
            if not is_variation_signal_numeric:
                variation_analysis_df['signal_binary'] = (variation_analysis_df[variation_signal_col] == variation_signal_value).astype(int)
            else:
                variation_analysis_df['signal_binary'] = variation_analysis_df[variation_signal_col]
            
            # Run analysis
            with st.spinner("Running comparative analysis..."):
                # Calculate key metrics
                
                # 1. Signal metrics
                control_signal_mean = control_analysis_df['signal_binary'].mean()
                variation_signal_mean = variation_analysis_df['signal_binary'].mean()
                
                # 2. Business metrics
                control_conversion = control_analysis_df['metric_binary'].mean()
                variation_conversion = variation_analysis_df['metric_binary'].mean()
                
                # 3. Calculate correlations
                try:
                    control_correlation, control_p_value = pearsonr(
                        control_analysis_df['signal_binary'], 
                        control_analysis_df['metric_binary']
                    )
                except:
                    control_correlation, control_p_value = 0, 1
                    
                try:
                    variation_correlation, variation_p_value = pearsonr(
                        variation_analysis_df['signal_binary'], 
                        variation_analysis_df['metric_binary']
                    )
                except:
                    variation_correlation, variation_p_value = 0, 1
                
                # 4. Statistical tests for differences
                
                # Signal difference
                signal_diff = variation_signal_mean - control_signal_mean
                signal_rel_diff = signal_diff / control_signal_mean if control_signal_mean != 0 else 0
                
                # Conversion difference
                conversion_diff = variation_conversion - control_conversion
                conversion_rel_diff = conversion_diff / control_conversion if control_conversion != 0 else 0
                
                # Z-test for conversion difference
                n1 = len(control_analysis_df)
                n2 = len(variation_analysis_df)
                p1 = control_conversion
                p2 = variation_conversion
                
                # Pooled proportion for z-test
                p_pooled = ((n1 * p1) + (n2 * p2)) / (n1 + n2)
                se = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
                
                if se > 0:
                    z_score = (p2 - p1) / se
                    conversion_p_value = 2 * (1 - stats.norm.cdf(abs(z_score)))
                else:
                    z_score = 0
                    conversion_p_value = 1
                
                # T-test for signal difference (if numeric)
                if is_control_signal_numeric and is_variation_signal_numeric:
                    signal_t_stat, signal_p_value = stats.ttest_ind(
                        control_analysis_df['signal_binary'].dropna(),
                        variation_analysis_df['signal_binary'].dropna(),
                        equal_var=False  # Using Welch's t-test (doesn't assume equal variances)
                    )
                else:
                    # Chi-square test for categorical signals
                    # Create contingency table
                    control_signal_positive = control_analysis_df['signal_binary'].sum()
                    control_signal_negative = len(control_analysis_df) - control_signal_positive
                    
                    variation_signal_positive = variation_analysis_df['signal_binary'].sum()
                    variation_signal_negative = len(variation_analysis_df) - variation_signal_positive
                    
                    contingency = np.array([
                        [control_signal_positive, control_signal_negative],
                        [variation_signal_positive, variation_signal_negative]
                    ])
                    
                    _, signal_p_value, _, _ = stats.chi2_contingency(contingency)
                
                # Correlation difference significance
                # Fisher's z-transformation to test difference between correlations
                z1 = np.arctanh(control_correlation)
                z2 = np.arctanh(variation_correlation)
                
                # Standard error
                se_diff = np.sqrt(1/(n1-3) + 1/(n2-3))
                
                # Z-score
                z_corr = (z2 - z1) / se_diff
                
                # P-value
                corr_diff_p_value = 2 * (1 - stats.norm.cdf(abs(z_corr)))
                
                # Store results in session state
                st.session_state.comparative_results = {
                    'control_df': control_analysis_df,
                    'variation_df': variation_analysis_df,
                    'control_signal_col': control_signal_col,
                    'variation_signal_col': variation_signal_col,
                    'control_metric_col': control_metric_col,
                    'variation_metric_col': variation_metric_col,
                    'is_control_signal_numeric': is_control_signal_numeric,
                    'is_variation_signal_numeric': is_variation_signal_numeric,
                    'control_signal_mean': control_signal_mean,
                    'variation_signal_mean': variation_signal_mean,
                    'control_conversion': control_conversion,
                    'variation_conversion': variation_conversion,
                    'control_correlation': control_correlation,
                    'variation_correlation': variation_correlation,
                    'control_p_value': control_p_value,
                    'variation_p_value': variation_p_value,
                    'signal_diff': signal_diff,
                    'signal_rel_diff': signal_rel_diff,
                    'conversion_diff': conversion_diff,
                    'conversion_rel_diff': conversion_rel_diff,
                    'conversion_p_value': conversion_p_value,
                    'signal_p_value': signal_p_value,
                    'corr_diff_p_value': corr_diff_p_value,
                    'n1': n1,
                    'n2': n2
                }
                st.session_state.has_comparative_results = True
                
                st.success("Analysis complete!")
    
    # Display results if available
    if st.session_state.has_comparative_results:
        results = st.session_state.comparative_results
        
        # Set significance threshold
        significance_level = st.slider(
            "Select significance level (p-value threshold)",
            min_value=0.01,
            max_value=0.10,
            value=0.05,
            step=0.01,
            format="%.2f",
            key="comp_significance_slider"
        )
        alpha = significance_level
        
        st.header("Comparison Results")
        
        # Overview metrics in a nice table
        st.subheader("Overall Metrics Comparison")
        
        # Create a DataFrame for comparison
        comparison_data = {
            'Metric': [
                'Sample Size', 
                f"Signal: {results['control_signal_col']}", 
                'Conversion Rate', 
                'Signal-Conversion Correlation'
            ],
            'Control': [
                results['n1'],
                f"{results['control_signal_mean']:.4f}" if results['is_control_signal_numeric'] else f"{results['control_signal_mean']:.2%}",
                f"{results['control_conversion']:.2%}",
                f"{results['control_correlation']:.4f}"
            ],
            'Variation': [
                results['n2'],
                f"{results['variation_signal_mean']:.4f}" if results['is_variation_signal_numeric'] else f"{results['variation_signal_mean']:.2%}",
                f"{results['variation_conversion']:.2%}",
                f"{results['variation_correlation']:.4f}"
            ],
            'Difference': [
                f"{results['n2'] - results['n1']} ({(results['n2'] - results['n1'])/results['n1']:.1%})",
                f"{results['signal_diff']:.4f} ({results['signal_rel_diff']:.1%})" if results['is_control_signal_numeric'] else f"{results['signal_diff']:.2%} ({results['signal_rel_diff']:.1%})",
                f"{results['conversion_diff']:.2%} ({results['conversion_rel_diff']:.1%})",
                f"{results['variation_correlation'] - results['control_correlation']:.4f} ({(results['variation_correlation'] - results['control_correlation'])/abs(results['control_correlation']):.1%})" if results['control_correlation'] != 0 else f"{results['variation_correlation'] - results['control_correlation']:.4f}"
            ],
            'Significant': [
                "",
                "✓" if results['signal_p_value'] < alpha else "",
                "✓" if results['conversion_p_value'] < alpha else "",
                "✓" if results['corr_diff_p_value'] < alpha else ""
            ],
            'p-value': [
                "",
                f"{results['signal_p_value']:.4f}",
                f"{results['conversion_p_value']:.4f}",
                f"{results['corr_diff_p_value']:.4f}"
            ]
        }
        
        comparison_df = pd.DataFrame(comparison_data)
        st.dataframe(comparison_df, use_container_width=True)
        
        # Visualizations
        st.subheader("Visualizations")
        
        viz_col1, viz_col2 = st.columns(2)
        
        with viz_col1:
            st.write("Signal Distribution")
            
            # Create histograms for signal distributions
            if results['is_control_signal_numeric'] and results['is_variation_signal_numeric']:
                # Create histogram data
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Control histogram
                ax.hist(
                    results['control_df']['signal_binary'].dropna(), 
                    alpha=0.5, 
                    bins=20, 
                    label=f"Control (mean: {results['control_signal_mean']:.2f})"
                )
                
                # Variation histogram
                ax.hist(
                    results['variation_df']['signal_binary'].dropna(), 
                    alpha=0.5, 
                    bins=20, 
                    label=f"Variation (mean: {results['variation_signal_mean']:.2f})"
                )
                
                ax.set_xlabel('Signal Value')
                ax.set_ylabel('Frequency')
                ax.set_title(f"Distribution Comparison: {results['control_signal_col']}")
                ax.legend()
                
                # Display the plot
                st.pyplot(fig)
                
                # Add significance annotation
                if results['signal_p_value'] < alpha:
                    st.success(f"The difference in signal values is statistically significant (p = {results['signal_p_value']:.4f}).")
                else:
                    st.info(f"The difference in signal values is not statistically significant (p = {results['signal_p_value']:.4f}).")
            else:
                # For categorical signals, use bar charts
                control_prop = results['control_signal_mean']
                variation_prop = results['variation_signal_mean']
                
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.bar(['Control', 'Variation'], [control_prop, variation_prop], color=['blue', 'green'], alpha=0.7)
                ax.set_ylabel('Proportion')
                ax.set_title(f"Signal Proportion Comparison: {results['control_signal_col']}")
                
                # Add percentages on top of bars
                for i, v in enumerate([control_prop, variation_prop]):
                    ax.text(i, v + 0.01, f"{v:.1%}", ha='center')
                
                st.pyplot(fig)
                
                # Add significance annotation
                if results['signal_p_value'] < alpha:
                    st.success(f"The difference in signal proportions is statistically significant (p = {results['signal_p_value']:.4f}).")
                else:
                    st.info(f"The difference in signal proportions is not statistically significant (p = {results['signal_p_value']:.4f}).")
        
        with viz_col2:
            st.write("Conversion Rate Comparison")
            
            # Create bar chart for conversion rates
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.bar(['Control', 'Variation'], [results['control_conversion'], results['variation_conversion']], color=['blue', 'green'], alpha=0.7)
            ax.set_ylabel('Conversion Rate')
            ax.set_title('Conversion Rate Comparison')
            
            # Add percentages on top of bars
            for i, v in enumerate([results['control_conversion'], results['variation_conversion']]):
                ax.text(i, v + 0.01, f"{v:.1%}", ha='center')
            
            st.pyplot(fig)
            
            # Add significance annotation
            if results['conversion_p_value'] < alpha:
                st.success(f"The difference in conversion rates is statistically significant (p = {results['conversion_p_value']:.4f}).")
            else:
                st.info(f"The difference in conversion rates is not statistically significant (p = {results['conversion_p_value']:.4f}).")
        
        # Correlation visualizations
        st.subheader("Signal-Conversion Relationship")
        
        corr_col1, corr_col2 = st.columns(2)
        
        with corr_col1:
            st.write("Control Group Correlation")
            
            # Create scatter plot for control
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results['control_df']['signal_binary'], results['control_df']['metric_binary'], alpha=0.5)
            ax.set_xlabel(results['control_signal_col'])
            ax.set_ylabel(results['control_metric_col'])
            ax.set_title(f"Control: r = {results['control_correlation']:.4f} (p = {results['control_p_value']:.4f})")
            
            # Add trend line
            if results['is_control_signal_numeric']:
                z = np.polyfit(results['control_df']['signal_binary'].dropna(), results['control_df']['metric_binary'].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(results['control_df']['signal_binary'].dropna(), p(results['control_df']['signal_binary'].dropna()), "r--", alpha=0.7)
            
            st.pyplot(fig)
        
        with corr_col2:
            st.write("Variation Group Correlation")
            
            # Create scatter plot for variation
            fig, ax = plt.subplots(figsize=(10, 6))
            ax.scatter(results['variation_df']['signal_binary'], results['variation_df']['metric_binary'], alpha=0.5, color='green')
            ax.set_xlabel(results['variation_signal_col'])
            ax.set_ylabel(results['variation_metric_col'])
            ax.set_title(f"Variation: r = {results['variation_correlation']:.4f} (p = {results['variation_p_value']:.4f})")
            
            # Add trend line
            if results['is_variation_signal_numeric']:
                z = np.polyfit(results['variation_df']['signal_binary'].dropna(), results['variation_df']['metric_binary'].dropna(), 1)
                p = np.poly1d(z)
                ax.plot(results['variation_df']['signal_binary'].dropna(), p(results['variation_df']['signal_binary'].dropna()), "r--", alpha=0.7)
            
            st.pyplot(fig)
        
        # Correlation difference significance
        if results['corr_diff_p_value'] < alpha:
            st.success(f"The difference in correlation strength is statistically significant (p = {results['corr_diff_p_value']:.4f}).")
        else:
            st.info(f"The difference in correlation strength is not statistically significant (p = {results['corr_diff_p_value']:.4f}).")
        
        # Overall assessment
        st.subheader("Overall Assessment")
        
        # Determine if variation is better
        is_signal_better = results['signal_diff'] > 0 and results['signal_p_value'] < alpha
        is_conversion_better = results['conversion_diff'] > 0 and results['conversion_p_value'] < alpha
        is_correlation_better = (results['variation_correlation'] > results['control_correlation']) and results['corr_diff_p_value'] < alpha
        
        if is_conversion_better:
            st.success(f"✅ Variation shows significantly better conversion rates (+{results['conversion_diff']:.2%}, p = {results['conversion_p_value']:.4f}).")
        elif results['conversion_diff'] > 0:
            st.info(f"ℹ️ Variation shows higher conversion rates, but the difference is not statistically significant (+{results['conversion_diff']:.2%}, p = {results['conversion_p_value']:.4f}).")
        else:
            st.warning(f"⚠️ Variation does not show improvement in conversion rates ({results['conversion_diff']:.2%}, p = {results['conversion_p_value']:.4f}).")
        
        if is_signal_better:
            st.success(f"✅ Variation shows significantly better signal values (+{results['signal_diff']:.4f}, p = {results['signal_p_value']:.4f}).")
        elif results['signal_diff'] > 0:
            st.info(f"ℹ️ Variation shows higher signal values, but the difference is not statistically significant (+{results['signal_diff']:.4f}, p = {results['signal_p_value']:.4f}).")
        
        if is_correlation_better:
            st.success(f"✅ Signal-Conversion correlation is significantly stronger in Variation (Δr = +{results['variation_correlation'] - results['control_correlation']:.4f}, p = {results['corr_diff_p_value']:.4f}).")
        elif results['variation_correlation'] > results['control_correlation']:
            st.info(f"ℹ️ Signal-Conversion correlation is stronger in Variation, but the difference is not statistically significant (Δr = +{results['variation_correlation'] - results['control_correlation']:.4f}, p = {results['corr_diff_p_value']:.4f}).")
