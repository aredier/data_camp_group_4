import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go

from backend import ReviewApp

app = dash.Dash()
<<<<<<< HEAD
backend = ReviewApp("data/test_predicted.db")
#backend.build_data_base(unlabeled="data/data_unlabeled.csv", labeled="data/labeled_data.csv")
#backend.update_data_base("data/scraper.log")
backend._build_vocab(preprocess=True)


=======

backend = ReviewApp("backend/data/test_predicted.db")
backend._build_vocab(preprocess=True)


COLORS = ["rgba(221, 167, 123, 1)",
          "rgba(206, 182, 174, 1)",
          "rgba(179, 151, 143, 1)",
          "rgba(203, 204, 291, 1)",
          "rgba(161, 162, 137, 1)",
          "rgba(132, 134, 113, 1)",
          "rgba(166, 156, 153, 1)",
          "rgba(121, 109, 105, 1)",
          "rgba(78, 68, 67, 1)",
          "rgba(134, 129, 128, 1)",
          "rgba(114, 112, 116, 1)",
          "rgba(76, 74, 79, 1)",
          "rgba(55, 61, 32, 1)"
          "rgba(61, 59, 63, 1)"
          ]

ISSUE_NAMES = backend.issue_categories
>>>>>>> 9451431686bf9b25e0f5395da6264aecb4aa51b7
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


def compute_reviews_over_time():
    issue_df = backend.find_issues().drop(["issue", "sentence"], axis=1)
    issues_grouped_by_date = issue_df.groupby(["date_time"]).agg(sum)
    dates = issues_grouped_by_date.index.values.tolist()
    data = {}
    for date in dates:
        total = 0
        for col in ISSUE_NAMES:
            value = int(issues_grouped_by_date.loc[date, col])
            total += value
            try:
                data[col]["total"].append(total)
                data[col]["value"].append(value)
            except KeyError:
                data[col] = {"total": [total], "value": [value]}

    return go.Figure(data=[
        go.Scatter(
            x=dates,
            y=data[col]["total"],
            fill='tonexty',
            mode="none",
            name=col,
            fillcolor=COLORS[i],
            text=[str(i) for i in data[col]["value"]],
            hoverinfo="text"
        )
        for i, col in enumerate(ISSUE_NAMES)
    ])


# LAYOUT
#
#
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
    html.H3("issue type detected over time"),
    dcc.Graph(
        id = "reviews_over_time",
        figure = compute_reviews_over_time()
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
