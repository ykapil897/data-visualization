import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from shiny import App, ui, render, reactive
import shinyswatch  # For themes

# Load a dataset from seaborn
tips_df = sns.load_dataset("tips")
penguins_df = sns.load_dataset("penguins")
titanic_df = sns.load_dataset("titanic")
planets_df = sns.load_dataset("planets")
flights_df = sns.load_dataset("flights")

available_datasets = {
    "tips": tips_df,
    "penguins": penguins_df,
    "titanic": titanic_df, 
    "planets": planets_df,
    "flights": flights_df
}

# Define UI with modern layout using Shiny components
app_ui = ui.page_sidebar(
    ui.sidebar(
        ui.h3("Data Selection"),
        ui.input_select("dataset", "Choose a Dataset:", 
                      choices=list(available_datasets.keys())),
        ui.hr(),
        ui.h3("Column Selection"),
        ui.output_ui("column_selector_ui"),
        ui.hr(),
        ui.h3("Plot Settings"),
        ui.input_select("plot_type", "Plot Type:", 
                      choices=["Bar Plot", "Histogram", "Box Plot", "Violin Plot"]),
        ui.input_checkbox("show_table", "Show Data Table", True),
        width="25%"
    ),
    ui.h2("Interactive Data Visualization Explorer"),
    ui.row(
        ui.column(12, ui.h3("Data Summary"), ui.output_ui("summary_ui")),
    ),
    ui.row(
        ui.column(6, ui.h3("Data Visualization"), ui.output_plot("main_plot")),
        ui.column(6, ui.h3("Data Table"), ui.output_ui("table_ui"))
    ),
    ui.br(),
    ui.div(
        ui.p("Created with Shiny for Python | Data from Seaborn", style="text-align: center; color: grey;")
    ),
    title="Data Explorer App",
    # Use shinyswatch theme instead of theme_bootstrap
    theme=shinyswatch.theme.minty()
)

# Server logic
def server(input, output, session):
    # Reactive expression to get the selected dataset
    @reactive.Calc
    def get_dataset():
        return available_datasets[input.dataset()]

    # UI for column selector based on selected dataset
    @output
    @render.ui
    def column_selector_ui():
        df = get_dataset()
        
        # Get numerical and categorical columns
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
        categorical_cols = df.select_dtypes(include=['object', 'category', 'bool']).columns.tolist()
        
        return ui.div(
            ui.input_select("x_var", "X Variable:", 
                         choices=categorical_cols if categorical_cols else numerical_cols),
            ui.input_select("y_var", "Y Variable (for applicable plots):",
                         choices=numerical_cols) if numerical_cols else None,
            ui.input_slider("n_bins", "Number of Bins (for histogram):", 
                         min=5, max=50, value=20) if numerical_cols else None
        )
    
    # Data summary
    @output
    @render.ui
    def summary_ui():
        df = get_dataset()
        summary_stats = df.describe().round(2)
        return ui.div(
            ui.HTML(f"<strong>Dataset:</strong> {input.dataset()} - {len(df)} rows, {len(df.columns)} columns"),
            ui.HTML(summary_stats.to_html(classes="table table-striped table-sm")),
            style="font-size: 0.9rem; max-height: 300px; overflow-y: auto;"
        )
    
    # Table display
    @output
    @render.ui
    def table_ui():
        if not input.show_table():
            return ui.p("Table display is hidden")
        
        df = get_dataset()
        return ui.div(
            ui.HTML(df.head(10).to_html(
                classes="table table-striped table-sm", 
                index=False
            )),
            style="font-size: 0.9rem; max-height: 400px; overflow-y: auto;"
        )
    
    # Main visualization
    @output
    @render.plot
    def main_plot():
        df = get_dataset()
        x_var = input.x_var()
        plot_type = input.plot_type()
        
        # Create figure with appropriate size
        fig, ax = plt.subplots(figsize=(8, 6))
        
        if plot_type == "Bar Plot":
            if x_var in df.columns:
                sns.countplot(data=df, x=x_var, ax=ax)
                ax.set_title(f"Count of {x_var}")
                ax.set_ylabel("Count")
                plt.xticks(rotation=45, ha='right')
        
        elif plot_type == "Histogram":
            if x_var in df.columns and df[x_var].dtype in ['int64', 'float64']:
                sns.histplot(data=df, x=x_var, bins=input.n_bins(), kde=True, ax=ax)
                ax.set_title(f"Distribution of {x_var}")
                ax.set_ylabel("Count")
        
        elif plot_type == "Box Plot":
            if "y_var" in input:
                y_var = input.y_var()
                if x_var in df.columns and y_var in df.columns:
                    sns.boxplot(data=df, x=x_var, y=y_var, ax=ax)
                    ax.set_title(f"{y_var} by {x_var}")
                    plt.xticks(rotation=45, ha='right')
        
        elif plot_type == "Violin Plot":
            if "y_var" in input:
                y_var = input.y_var()
                if x_var in df.columns and y_var in df.columns:
                    sns.violinplot(data=df, x=x_var, y=y_var, ax=ax)
                    ax.set_title(f"{y_var} by {x_var}")
                    plt.xticks(rotation=45, ha='right')
        
        plt.tight_layout()
        return fig

# Create the app
app = App(app_ui, server)