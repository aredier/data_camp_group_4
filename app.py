import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from backend import ReviewApp

app = dash.Dash()
backend = ReviewApp("data/test_predicted.db")
#backend.build_data_base(unlabeled="data/data_unlabeled.csv", labeled="data/labeled_data.csv")
#backend.update_data_base("data/scraper.log")
backend._build_vocab(preprocess=True)


def compute_issue_phone_pie_chart():
    issue_count = [(k,v) for k, v in backend._base.get_phone_issue_count(True).items()]
    return {
        "data": [go.Pie(
            labels=[x[0] for x in issue_count],
            values=[x[1] for x in issue_count]
        )]
    }

def rebuild_issues():
    issue_df = backend.find_issues()
    def build_single_issue(row):
        return {"sentence" : row["sentence"],
         "issues" : row[row==1].index.tolist()
         }
    return issue_df.apply(build_single_issue, axis=1).to_dict()


app.layout = html.Div([
    html.Div([
        html.H2("training model"),
        html.Div([
            dcc.Dropdown(
                id = "train_dopdown",
                options = [
                    {"label" : "XGBoost (recomended)", "value" : "xgb"},
                    {"label" : "Random Forest", "value" : "rf"},
                    {"label" : "logistic regression", "value" : "logreg"}
                ],
                value = "xgb"
            ),
            html.Button("train", id="train_button"),
            html.Div(id="train_resume")
        ],
        style = {"display" : "inline"}
        ),
    ]),
    dcc.Checklist(
        id = "issue_type_source",
        options = [
            {"label" : "include predicted by model" , "value" : "predicted"}
        ],
        values = []
    ),
    dcc.Graph(id="issue_type_graph"),
    html.H1("test"),
    dcc.Graph(
        id="issue_phone_graph",
        figure = compute_issue_phone_pie_chart()
    ),
    dcc.Dropdown(
            id='categories',
            options=[{'label': i, 'value': i} for i in backend.issue_categories],
            multi=True
    ),
    html.Div(id ="new_issue_list")
])

@app.callback(
    Output("issue_type_graph", "figure"),
    [Input("issue_type_source", "values")]
)
def compute_issue_type_pie_chart(options):
    issue_count =[ (k, v) for k, v in backend.issue_type_count("predicted" in options).items()]
    return {
        "data" : [go.Pie(
            labels = [x[0] for x in issue_count],
            values = [x[1] for x in issue_count]
        )]
    }

@app.callback(
    Output("new_issue_list", "children"),
    [
        Input("categories", "value")
    ]
)
def get_new_issues(categories):
    if categories is None or categories == list():
        categories = ["issue"]
    return [
        html.Div([
            html.H3(", ".join([issue for issue in issue_dic["issues"] if issue != "issue"])),
            html.P(issue_dic["sentence"])
        ])

        for k, issue_dic
        in rebuild_issues().items()
        if set(issue_dic["issues"]) & set(categories) != set()
    ]

@app.callback(
    Output("train_resume", "children"),
    [
        Input("train_button", "n_clicks")
    ],
    [
        State("train_dopdown", "value")
    ]
)

def train_backend_and_return_resume(clicks, model):
    if clicks is None:
        clicks = 0
    if clicks == 1:
        return [
            html.P(str(value))
            for value in backend.train_model(model, return_test_analysis=True)
        ]


        return True
    elif clicks > 1:
        return [
            html.P(str(value))
            for value in backend.retrain(model, return_test_analysis=True)
        ]

if __name__ == '__main__':
    app.run_server()
