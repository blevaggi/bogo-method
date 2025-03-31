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


st.title("The BoGo Mindset")
st.subheader("An app for correlation analyses between signals and business outcomes")
st.write("Upload a file to analyze the relationship between a signal column and a business metric.")
st.write("Examples of signals could be: an eval metric, a label like Urgent/Non-urgent, the character count of a user's first message, the theme label of the chat, etc.")

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
