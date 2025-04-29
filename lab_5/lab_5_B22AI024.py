import pandas as pd
import shinyswatch
import matplotlib.pyplot as plt
import seaborn as sns
from shiny import App, ui, render, reactive

# Define UI with a sidebar layout
app_ui = ui.page_fluid(
    ui.panel_title("CSV Data Explorer"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.h3("Data Controls"),
            ui.input_file("file", "Upload CSV", accept=[".csv"]),
            ui.hr(),
            ui.h3("Column Selection"),
            ui.output_ui("column_selector_ui"),
            ui.hr(),
            ui.input_slider("display_limit", "Maximum unique values to show:", 
                        min=3, max=50, value=10),
            ui.hr(),
            ui.h3("Plot Settings"),
            ui.input_select("plot_type", "Plot Type:", 
                          choices=["Bar Plot", "Histogram", "Box Plot", "Violin Plot", "Count Plot"]),
            ui.input_checkbox("show_filters", "Show Data Filters", True),

            ui.input_action_button(
                "refresh_viz", 
                ui.tags.span(ui.tags.i(class_="fa fa-sync"), " Update Visualization"),
                class_="btn-primary btn-lg mt-3 w-100",
                style="font-weight: bold;"
            ),
        ),
        
        # Main panel with two columns
        ui.row(
            # Left column for the plot
            ui.column(6, 
                ui.card(
                    ui.card_header("Data Visualization"),
                    ui.output_plot("category_plot")
                )
            ),
            # Right column for the table
            ui.column(6, 
                ui.card(
                    ui.card_header("Data Table"),
                    ui.output_ui("data_table_ui")
                )
            )
        ),
        
        # Summary stats at the bottom
        ui.row(
            ui.column(12, 
                ui.card(
                    ui.card_header("Summary Statistics"),
                    ui.output_ui("summary_ui")
                )
            )
        )
    ),
    theme=shinyswatch.theme.minty()
)

# Server logic
def server(input, output, session):
    # Reactive expression for filtered data
    @reactive.Calc
    def filtered_data():
        # Explicitly track reactivity to refresh button
        input.refresh_viz()
        
        file = input.file()
        if not file:
            return pd.DataFrame()
        
        # Debug - print when this function is called
        print("filtered_data() being recalculated")
        
        try:
            # Base data from file with error handling
            df = pd.read_csv(file[0]['datapath'], encoding="utf-8", on_bad_lines='skip')
            print(f"Original data has {len(df)} rows")
            
            # Try a simpler approach using filtered_table_filter
            if hasattr(input, "filtered_table_filter") and callable(getattr(input, "filtered_table_filter", None)):
                try:
                    # Get the filter conditions
                    filter_conditions = input.filtered_table_filter()
                    print(f"Filter conditions: {filter_conditions}")
                    
                    # Apply filters manually if they exist
                    if filter_conditions:
                        filtered_df = df.copy()
                        
                        # Handle tuple format (what we're seeing in the logs)
                        if isinstance(filter_conditions, tuple):
                            for condition in filter_conditions:
                                if 'col' in condition and 'value' in condition:
                                    col_idx = condition['col']
                                    # Convert column index to name
                                    col_name = df.columns[col_idx] if isinstance(col_idx, int) else col_idx
                                    
                                    # Handle range filter (min, max tuple)
                                    if isinstance(condition['value'], tuple):
                                        min_val, max_val = condition['value']
                                        if min_val is not None:
                                            filtered_df = filtered_df[filtered_df[col_name] >= min_val]
                                        if max_val is not None:
                                            filtered_df = filtered_df[filtered_df[col_name] <= max_val]
                                            
                                    elif isinstance(condition['value'], str):
                                        # For string values, use partial matching (contains)
                                        if pd.api.types.is_object_dtype(df[col_name]):
                                            # Case-insensitive partial matching
                                            filtered_df = filtered_df[filtered_df[col_name].str.contains(condition['value'], case=False, na=False)]
                                        else:
                                            # For non-string columns, convert to string for matching
                                            filtered_df = filtered_df[filtered_df[col_name].astype(str).str.contains(condition['value'], case=False, na=False)]

                                    # Handle other value types
                                    elif isinstance(condition['value'], list):
                                        filtered_df = filtered_df[filtered_df[col_name].isin(condition['value'])]
                                    else:
                                        filtered_df = filtered_df[filtered_df[col_name] == condition['value']]
                            
                            print(f"Applied tuple-style filtering, got {len(filtered_df)} rows")
                            return filtered_df
                        
                        # Original dictionary-style handling (keep as fallback)
                        elif isinstance(filter_conditions, dict):
                            for col, condition in filter_conditions.items():
                                if "value" in condition:
                                    if isinstance(condition["value"], list):
                                        # For multiple values (in/not in)
                                        if condition["op"] == "in":
                                            filtered_df = filtered_df[filtered_df[col].isin(condition["value"])]
                                        elif condition["op"] == "not in":
                                            filtered_df = filtered_df[~filtered_df[col].isin(condition["value"])]
                                    else:
                                        # For single value comparisons
                                        if condition["op"] == "==":
                                            filtered_df = filtered_df[filtered_df[col] == condition["value"]]
                                        elif condition["op"] == "!=":
                                            filtered_df = filtered_df[filtered_df[col] != condition["value"]]
                                        elif condition["op"] == ">":
                                            filtered_df = filtered_df[filtered_df[col] > condition["value"]]
                                        elif condition["op"] == ">=":
                                            filtered_df = filtered_df[filtered_df[col] >= condition["value"]]
                                        elif condition["op"] == "<":
                                            filtered_df = filtered_df[filtered_df[col] < condition["value"]]
                                        elif condition["op"] == "<=":
                                            filtered_df = filtered_df[filtered_df[col] <= condition["value"]]
                            
                            print(f"Applied manual filtering, got {len(filtered_df)} rows")
                            return filtered_df
                except Exception as e:
                    print(f"Error applying manual filters: {e}")
            
            # If we're here, try the data_view approach
            if hasattr(input, "filtered_table_data_view_rows"):
                try:
                    # Get the indices of visible rows
                    visible_rows = input.filtered_table_data_view_rows()
                    
                    # More robust handling for visible_rows
                    if visible_rows is not None:
                        print(f"Visible rows type: {type(visible_rows)}")
                        
                        # Handle single value case
                        if not hasattr(visible_rows, '__iter__'):
                            # Single row case
                            visible_rows = [int(visible_rows)]
                        else:
                            # Try to convert to list of integers
                            try:
                                visible_rows = [int(r) for r in visible_rows]
                            except:
                                # If we can't convert directly, skip this approach
                                print("Could not convert visible_rows to integers")
                                raise ValueError("Invalid visible_rows format")
                        
                        # Only filter if we have fewer rows than the original
                        if len(visible_rows) < len(df):
                            try:
                                filtered_df = df.iloc[visible_rows].copy()
                                print(f"Using data view rows filtering: {len(filtered_df)} rows")
                                return filtered_df
                            except Exception as e:
                                print(f"Error subsetting with visible rows: {e}")
                except Exception as e:
                    print(f"Error using data_view_rows: {e}")
                        
        except Exception as e:
            import traceback
            print(f"Error reading CSV: {e}")
            print(traceback.format_exc())
            return pd.DataFrame()
                
        # If all filtering attempts fail, return the original dataframe
        print("Using original dataframe (no filtering applied)")
        return df
    
    # Data table display
    @output
    @render.ui
    def data_table_ui():
        file = input.file()
        if not file:
            return ui.p("Upload a CSV file to see data.")
        
        # Show filters only if requested
        if not input.show_filters():
            # Simple table without filters
            df = filtered_data()
            return ui.div(
                ui.HTML(df.head(10).to_html(
                    classes="table table-striped table-sm",
                    index=False
                )),
                style="font-size: 0.9rem; max-height: 400px; overflow-y: auto;"
            )
        
        # Create a filterable table with clearer instructions
        return ui.div(
            ui.p(
                ui.tags.span(ui.tags.i(class_="fa fa-filter"), " "),
                ui.tags.span("Use column filters to subset data, then "),
                ui.tags.strong("click 'Refresh Visualization'"),
                style="font-size: 0.9rem; color: #555;"
            ),
            ui.output_data_frame("filtered_table"),
            style="font-size: 0.9rem; max-height: 500px; overflow-y: auto;"
        )

    @output
    @render.data_frame
    def filtered_table():
        file = input.file()
        if not file:
            return pd.DataFrame()
        
        df = pd.read_csv(file[0]['datapath'], encoding="utf-8")
        
        # Return as DataGrid with column filtering enabled
        return render.DataGrid(
            df, 
            filters=True, 
            height="400px"
        )
        
    # Summary statistics based on filtered data
    @output
    @render.ui
    def summary_ui():
        file = input.file()
        if not file:
            return ui.p("Upload a CSV file to see summary statistics.")
        
        # Use filtered data for summary
        df = filtered_data()
        if df.empty:
            return ui.p("No data available after filtering.")
        
        # Compare with original data to show filter effect
        try:
            original_df = pd.read_csv(file[0]['datapath'], encoding="utf-8", on_bad_lines='skip')
            filter_message = f"<span style='color:#0066cc;'><strong>Showing {len(df)} of {len(original_df)} rows ({round(len(df)/len(original_df)*100, 1)}% of data)</strong></span>"
        except:
            filter_message = f"<strong>Dataset Summary:</strong> {len(df)} rows, {len(df.columns)} columns"
            
        summary_stats = df.describe().round(2)
        return ui.div(
            ui.HTML(filter_message),
            ui.HTML(summary_stats.to_html(classes="table table-striped table-sm")),
            style="font-size: 0.9rem; max-height: 300px; overflow-y: auto;"
        )

    # Column selector for visualization with plot type
    @output
    @render.ui
    def column_selector_ui():
        file = input.file()
        if not file:
            return ui.p("Upload a CSV file to select columns.")

        df = filtered_data()
        if df.empty:
            return ui.p("No data available. Please check your CSV file.")
        
        # More robust numerical and categorical detection
        try:
            numerical_cols = []
            categorical_cols = []
            
            for col in df.columns:
                if df[col].dtype in ['int64', 'float64', 'int32', 'float32']:
                    numerical_cols.append(col)
                elif df[col].dtype in ['object', 'category', 'bool']:
                    categorical_cols.append(col)
                else:
                    # Try to determine type by content
                    try:
                        pd.to_numeric(df[col])
                        numerical_cols.append(col)
                    except:
                        if df[col].nunique() < min(20, len(df) // 2):
                            categorical_cols.append(col)
                        else:
                            numerical_cols.append(col)
        except Exception as e:
            # Fallback to simple approach
            numerical_cols = [col for col in df.columns if df[col].nunique() > 20]
            categorical_cols = [col for col in df.columns if col not in numerical_cols]
        
        # Get the current plot type
        plot_type = input.plot_type()
        
        # For plots that need both X and Y variables
        if plot_type in ["Box Plot", "Violin Plot"]:
            return ui.div(
                ui.input_select("x_var", "X Variable (Categories):", 
                            choices=categorical_cols if categorical_cols else df.columns.tolist()),
                ui.input_select("y_var", "Y Variable (Values):", 
                            choices=numerical_cols if numerical_cols else df.columns.tolist())
            )
        
        # For histograms (numerical X)
        elif plot_type == "Histogram":
            if not numerical_cols:
                return ui.div(
                    ui.p("Histogram works best with numerical columns."),
                    ui.input_select("x_var", "Variable to plot:", choices=df.columns.tolist()),
                    ui.input_slider("n_bins", "Number of bins:", min=5, max=50, value=20),
                    ui.input_checkbox("show_kde", "Show density curve", True)
                )
            
            return ui.div(
                ui.input_select("x_var", "Variable to plot:", choices=numerical_cols),
                ui.input_slider("n_bins", "Number of bins:", min=5, max=50, value=20),
                ui.input_checkbox("show_kde", "Show density curve", True)
            )
        
        # For bar plots and count plots (categorical X)
        else:  # Bar Plot, Count Plot
            
            return ui.div(
                ui.input_select("x_var", "Variable to plot:", 
                            choices=categorical_cols if categorical_cols else df.columns.tolist()),
                ui.input_checkbox("sort_values", "Sort by frequency", True)
            )
        
    # Visualization with multiple plot types
    @output
    @render.plot
    def category_plot():
        file = input.file()
        if not file:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Upload a CSV file to start", 
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Get the filtered data
        df = filtered_data()
        if df.empty:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "No data available", 
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Make sure input selections exist
        if not hasattr(input, "x_var"):
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, "Please select columns to visualize", 
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        x_var = input.x_var()
        plot_type = input.plot_type()
        
        if x_var not in df.columns:
            fig, ax = plt.subplots(figsize=(8, 6))
            ax.text(0.5, 0.5, f"Column '{x_var}' not found", 
                ha='center', va='center', fontsize=14)
            ax.axis('off')
            return fig
        
        # Create figure
        fig, ax = plt.subplots(figsize=(10, 6))
        
        try:
            # Using a more direct approach like in app.py
            if plot_type == "Bar Plot":
                # Convert to categorical if needed
                if not pd.api.types.is_categorical_dtype(df[x_var]) and not pd.api.types.is_object_dtype(df[x_var]):
                    df[x_var] = df[x_var].astype(str)
                
                # Get value counts
                value_counts = df[x_var].value_counts()
                if hasattr(input, "sort_values") and input.sort_values():
                    value_counts = value_counts.sort_values(ascending=False)
                
                # Limit categories
                limit = input.display_limit() 
                value_counts = value_counts.head(limit)
                
                # Create bar plot directly
                sns.barplot(x=value_counts.index, y=value_counts.values, ax=ax)
                ax.set_title(f"Distribution of '{x_var}' (Filtered: {len(df)} rows)")            
                ax.set_ylabel("Count")
                
                # Add value labels
                for i, v in enumerate(value_counts.values):
                    ax.text(i, v + 0.1, str(v), ha='center')
            
            elif plot_type == "Histogram":
                try:
                    # Try to convert to numeric if not already
                    if not pd.api.types.is_numeric_dtype(df[x_var]):
                        df[x_var] = pd.to_numeric(df[x_var])
                    
                    bins = input.n_bins() if hasattr(input, "n_bins") else 20
                    kde = input.show_kde() if hasattr(input, "show_kde") else True
                    
                    sns.histplot(data=df, x=x_var, bins=bins, kde=kde, ax=ax)
                    ax.set_title(f"Histogram of '{x_var}' (Filtered: {len(df)} rows)")                    
                    ax.set_ylabel("Count")
                except:
                    ax.text(0.5, 0.5, "Cannot create histogram with this data", 
                        ha='center', va='center', fontsize=14)
                    ax.axis('off')
            
            elif plot_type == "Box Plot":
                if hasattr(input, "y_var"):
                    y_var = input.y_var()
                    if y_var in df.columns:
                        # Convert x to categorical if needed
                        if not pd.api.types.is_categorical_dtype(df[x_var]) and not pd.api.types.is_object_dtype(df[x_var]):
                            df[x_var] = df[x_var].astype(str)
                            
                        # Limit categories
                        if df[x_var].nunique() > input.display_limit():
                            top_cats = df[x_var].value_counts().head(input.display_limit()).index
                            plot_df = df[df[x_var].isin(top_cats)]
                        else:
                            plot_df = df
                            
                        sns.boxplot(data=plot_df, x=x_var, y=y_var, ax=ax)
                        ax.set_title(f"{y_var} by {x_var} (Filtered: {len(df)} rows)")
                else:
                    sns.boxplot(data=df, y=x_var, ax=ax)
                    ax.set_title(f"Box Plot of {x_var} (Filtered: {len(df)} rows)")            

            elif plot_type == "Count Plot":
                # Convert to categorical if needed
                if not pd.api.types.is_categorical_dtype(df[x_var]) and not pd.api.types.is_object_dtype(df[x_var]):
                    df[x_var] = df[x_var].astype(str)
                    
                # Get order based on frequency
                limit = input.display_limit()
                if hasattr(input, "sort_values") and input.sort_values():
                    order = df[x_var].value_counts().head(limit).index
                else:
                    order = df[x_var].value_counts().head(limit).sort_index().index
                    
                # Simple countplot like in the tutorial
                sns.countplot(data=df, x=x_var, order=order, ax=ax)
                ax.set_title(f"Count of {x_var} (Filtered: {len(df)} rows)")                
                ax.set_ylabel("Count")
                
                # Add count labels
                for p in ax.patches:
                    ax.annotate(f'{int(p.get_height())}', 
                            (p.get_x() + p.get_width() / 2., p.get_height()), 
                            ha = 'center', va = 'bottom')
            
            elif plot_type == "Violin Plot":
                if hasattr(input, "y_var"):
                    y_var = input.y_var()
                    if y_var in df.columns:
                        # Convert x to categorical if needed
                        if not pd.api.types.is_categorical_dtype(df[x_var]) and not pd.api.types.is_object_dtype(df[x_var]):
                            df[x_var] = df[x_var].astype(str)
                            
                        # Limit categories
                        if df[x_var].nunique() > input.display_limit():
                            top_cats = df[x_var].value_counts().head(input.display_limit()).index
                            plot_df = df[df[x_var].isin(top_cats)]
                        else:
                            plot_df = df
                            
                        sns.violinplot(data=plot_df, x=x_var, y=y_var, ax=ax)
                        ax.set_title(f"{y_var} by {x_var} (Filtered: {len(df)} rows)")            

            # Format x-axis labels for better readability
            plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
        except Exception as e:
            # Clear any partial plot
            ax.clear()
            ax.text(0.5, 0.5, f"Error creating plot: {str(e)}", 
                ha='center', va='center', fontsize=12)
            ax.axis('off')
        
        return fig

# Create the app
app = App(app_ui, server)