import shiny
from shiny import App, render, ui, reactive
import pandas as pd
import plotly.express as px
import requests
import json
import csv
import io
import os
from typing import List

# Internal API key (replace with your actual API key)
OPENAI_API_KEY = os.environ.get("OPENAI_API_KEY")

app_ui = ui.page_fluid(
    ui.panel_title("Fake Dataset Generator"),
    ui.layout_sidebar(
        ui.sidebar(
            ui.input_text("description", "Describe the dataset you want", 
                          placeholder="e.g., health data for a family of 4"),
            ui.input_action_button("generate", "Generate Dataset"),
            ui.download_button("download", "Download CSV"),
            ui.output_ui("summary")
        ),
        ui.navset_tab(
            ui.nav_panel("Data Table", ui.output_data_frame("dataset")),
            ui.nav_panel("Visualizations",
                ui.input_select("variable", "Select Variable", choices=[]),
                ui.output_plot("plot")
            )
        )
    )
)

def server(input, output, session):
    dataset = reactive.Value(None)
    summary_text = reactive.Value("")

    def preprocess_csv(csv_string: str) -> pd.DataFrame:
        csv_io = io.StringIO(csv_string)
        df = pd.read_csv(csv_io)
        df = df.apply(lambda x: pd.to_numeric(x, errors='ignore'))
        return df

    def generate_summary(df: pd.DataFrame) -> str:
        prompt = f"""Summarize the following dataset:

        Dimensions: {df.shape[0]} rows and {df.shape[1]} columns

        Variables:
        {', '.join(df.columns)}

        Please provide a brief summary of the dataset dimensions and variable definitions. Keep it concise, about 3-4 sentences."""

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that summarizes datasets."},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code == 200:
            content = response.json()
            summary = content['choices'][0]['message']['content']
            return summary
        else:
            return "Error generating summary. Please try again later."

    @reactive.Effect
    @reactive.event(input.generate)
    def _():
        description = input.description()
        if not description:
            return

        prompt = f"Generate a fake dataset as a CSV string based on this description: {description} Include a header row. Limit to 25 rows of data. Ensure all rows have the same number of columns. Do not include any additional text or explanations."

        response = requests.post(
            "https://api.openai.com/v1/chat/completions",
            headers={"Authorization": f"Bearer {OPENAI_API_KEY}"},
            json={
                "model": "gpt-3.5-turbo-0125",
                "messages": [
                    {"role": "system", "content": "You are a helpful assistant that generates fake datasets."},
                    {"role": "user", "content": prompt}
                ]
            }
        )

        if response.status_code == 200:
            content = response.json()
            csv_string = content['choices'][0]['message']['content']

            try:
                df = preprocess_csv(csv_string)
                dataset.set(df)
                ui.update_select("variable", choices=df.columns.tolist())

                # Generate and set summary
                summary = generate_summary(df)
                summary_text.set(summary)

            except Exception as e:
                ui.notification_show(f"Error parsing CSV: {str(e)}", type="error")
        else:
            ui.notification_show("Error generating dataset. Please try again later.", type="error")

    @output
    @render.data_frame
    def dataset():
        if dataset.get() is not None:
            return render.DataGrid(dataset.get())

    @output
    @render.plot
    def plot():
        df = dataset.get()
        if df is None or input.variable() is None:
            return None

        var = input.variable()
        if pd.api.types.is_numeric_dtype(df[var]):
            fig = px.histogram(df, x=var, title=f"Histogram of {var}")
        else:
            df_count = df[var].value_counts().nlargest(10).reset_index()
            df_count.columns = [var, 'count']
            fig = px.bar(df_count, x=var, y='count', title="Column Chart")
            fig.update_layout(xaxis_title=None, yaxis_title=None)

        return fig

    @output
    @render.ui
    def summary():
        if summary_text.get():
            return ui.div(
                ui.h4("Dataset Summary"),
                ui.p(summary_text.get()),
                {"style": "background-color: #f0f0f0; padding: 10px; border-radius: 5px;"}
            )

    @session.download(filename="generated_dataset.csv")
    def download():
        if dataset.get() is not None:
            return dataset.get().to_csv(index=False)

app = App(app_ui, server)