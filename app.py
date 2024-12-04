
from shiny import App, ui, render
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import accuracy_score, confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

# App UI
app_ui = ui.page_fluid(
    ui.navset_tab(
        ui.nav_panel("Overview", ui.h2("Project Overview"), ui.p("Description of the project.")),
        ui.nav_panel("Visualizations", ui.h2("Visualizations"), ui.p("Add plots or visualizations here.")),
        ui.nav_panel(
            "Models",
            ui.h2("Random Forest Model"),
            ui.output_text("model_accuracy"),
            ui.output_plot("conf_matrix"),
            ui.output_plot("feature_importance"),
        ),
        ui.nav_panel(
            "Data Table",
            ui.h2("Interactive Data Table"),
            ui.output_ui("data_table"),  # Render the table as HTML
        ),
    )
)

# Server Logic
def server(input, output, session):
    # Load dataset
    data = pd.read_csv('Merged_Data 1.csv')  # Replace with your actual dataset path

    # Preprocess data
    data['Years of SPD Service'] = data['Years of SPD Service'].replace(['< 1', '<1'], 0.5).astype(float)
    data_filtered = data[['Officer Disciplined?', 'Years of SPD Service', 'Subject Race', 'Subject Gender', 'Subject Age', 'Fatal', 'Disposition']].dropna()
    X = data_filtered[['Officer Disciplined?', 'Years of SPD Service', 'Subject Race', 'Subject Gender', 'Subject Age', 'Disposition']]
    y = data_filtered['Fatal'].apply(lambda x: 1 if x == 'Yes' else 0)

    # Define preprocessing and model pipeline
    categorical_features = ['Officer Disciplined?', 'Subject Race', 'Subject Gender', 'Disposition']
    preprocessor = ColumnTransformer(
        transformers=[('cat', OneHotEncoder(drop='first'), categorical_features)],
        remainder='passthrough'
    )
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('classifier', RandomForestClassifier(random_state=42))
    ])

    # Train/Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # Outputs
    @output
    @render.text
    def model_accuracy():
        return f"Model Accuracy: {accuracy:.2f}"

    @output
    @render.plot
    def conf_matrix():
        cm = confusion_matrix(y_test, y_pred)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=['Not Fatal', 'Fatal'])
        disp.plot(cmap='Blues')
        plt.title("Confusion Matrix")
        return plt.gcf()

    @output
    @render.plot
    def feature_importance():
        feature_importances = pipeline.named_steps['classifier'].feature_importances_
        encoded_feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        plt.figure(figsize=(10, 6))
        plt.barh(range(len(feature_importances)), feature_importances, align='center')
        plt.yticks(range(len(feature_importances)), encoded_feature_names)
        plt.xlabel('Feature Importance')
        plt.ylabel('Feature')
        plt.title('Feature Importances')
        plt.tight_layout()
        return plt.gcf()

    @output
    @render.ui
    def data_table():
        # Convert the data to an HTML table
        return ui.HTML(data.head(10).to_html(index=False))

# App
app = App(app_ui, server)

# Run App
if __name__ == "__main__":
    from shiny import run_app
    run_app(app)
