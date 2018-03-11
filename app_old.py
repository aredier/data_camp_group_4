import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import arrow

from backend import ReviewApp

app = dash.Dash()
app.config['suppress_callback_exceptions']=True

backend = ReviewApp("data/test_predicted.db")
#backend.build_data_base(unlabeled="data/data_unlabeled.csv", labeled="data/labeled_data.csv")
#backend.update_data_base("data/scraper.log")
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
DISPLAYED_ISSUES = []


class IssueCallBackGenerator:

    def __init__(self):
        self.issue_ids = []

    def clean_callback_list(self):
        self.issue_ids = []

    def process_isse(self,id):
        self.issue_ids.append(id)
        return id

    @staticmethod
    def basic_callbacks(clicks, values):
        if clicks > 0 :
            print("test")
            return "changed issue to {}".format(" ,".join(values))

    @staticmethod
    def general_callback(*children):
        print("got there")
        return children[0]

    def build_issue_change_callbacks(self):
        for id in self.issue_ids:
            app.callback(
                Output("issue_{}_callback_output".format(id), "children"),
                [Input("update_database", "n_clicks")],
                [State("issue_{}".format(id), "value")]
            )(self.basic_callbacks)
            print("callback id {} created ".format(id))

        app.callback(
            Output("change_issue_message", "children"),
            [
                Input("issue_{}_callback_output".format(id), "children")
                for id in self.issue_ids
            ]
        )(self.general_callback)
        print("general callback created")

issue_call_backs = IssueCallBackGenerator()


def compute_issue_phone_pie_chart():
    issue_count = [(k,v) for k, v in backend._base.get_phone_issue_count(True).items()]
    return {
        "data": [go.Pie(
            labels=[x[0] for x in issue_count],
            values=[x[1] for x in issue_count]
        )]
    }


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
            dcc.Checklist(
                id = "do_cv",
                options = [
                    {"label" : "do cross validation" , "value" : "cv"}
                ],
                values = []
            ),
            html.Button("update predictions (beware, very long the first time)", id="update_predictions_button"),
            html.P(id="update_error_message"),
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
    html.Div(id="issue_phone_graph"),
    html.H3("issue type detected over time"),
    html.Div(
        id = "reviews_over_time"
    ),
    html.P(id="change_issue_message",
           children="placeholder"),
    dcc.Dropdown(
            id='categories',
            options=[{'label': i, 'value': i} for i in ISSUE_NAMES],
            multi=True
    ),
    html.Button("update datebase", id="update_database"),
    html.P(id="invisible_text",
           children="placeholder "
           ),
    html.Div(id ="new_issue_list"),
])

@app.callback(
    Output("update_error_message", "children"),
    [
        Input("update_predictions_button", "n_clicks")
    ]
)
def update_predictions(clicks):
    if clicks is None:
        clicks = 0
    if clicks > 0:
        try:
            print("test")
            backend.update_predictions()
            return
        except AssertionError:
            print("raised exception")
            return "you can only update the predictions when the model is trained"

@app.callback(
    Output("issue_phone_graph", "children"),
    [
        Input("update_predictions_button", "n_clicks")
    ]
)
def return_issue_per_phone_graph(bs):
    if bs is None:
        bs = 0
    if backend.predicted and bs > 0:
        return dcc.Graph(id="phone_issue_graph", figure=compute_issue_phone_pie_chart())

    else:
        return html.P("sorry, this component can only be loaded once the predictions have been updated")

@app.callback(
    Output("reviews_over_time", "children"),
   [
       Input("update_predictions_button", "n_clicks"),
   ]
)
def compute_reviews_over_time(bs):
    if bs is None:
        bs = 0
    if backend.predicted and bs > 0:
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

        return dcc.Graph(id = "reviews_over_time_graph", figure = go.Figure(data=[
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
        ]))
    else:
        return html.P("sorry, this component can only be loaded once the predictions have been updated")

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

def rebuild_issues():
    issue_df = backend.find_issues()
    #internal function
    def build_single_issue(row):
        return({"id" : row["id"],
                "sentence" : row["sentence"],
                "issues" : row[row==1].index.tolist(),
                "date" : arrow.get(row["date_time"], "YYYY-MM-DD HH:mm:ss.000000").format("YYYY-MM-DD")
         })
    values = issue_df.apply(build_single_issue, axis=1).values.tolist()
    return values



@app.callback(
    Output("new_issue_list", "children"),
    [
        Input("categories", "value")
    ]
)
def get_new_issues(categories):
    issue_call_backs.clean_callback_list()
    if categories is None or categories == list():
        categories = ["issue"]
    return html.Table([
        html.Tr([
            html.Td(
                issue_dic["date"],
                style={
                    "border": "1px solid black",
                    "border-collapse": "collapse"
                }
            ),
            html.Td(issue_dic["sentence"],
                    style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        "width" : "50%"
                    }),
            html.Td(dcc.Dropdown(
                        id='issue_{}'.format(issue_dic["id"]),
                        options=[{'label': i, 'value': i} for i in ISSUE_NAMES],
                        multi=True,
                        value = issue_dic["issues"]
                     ),
                    style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        "width" : "0%"
                    })
        ])

        for issue_dic
        in rebuild_issues()
        if set(issue_dic["issues"]) & set(categories) != set()
    ], style = {
        "border" : "1px solid black",
        "border-collapse" : "collapse"
    })



@app.callback(
    Output("train_resume", "children"),
    [
        Input("train_button", "n_clicks")
    ],
    [
        State("train_dopdown", "value"),
        State("do_cv", "values")
    ]
)

def train_backend_and_return_resume(clicks, model, cv):
    if clicks is None:
        clicks = 0
    if clicks > 0:
        try:
            return html.Table([
                html.Tr([
                    html.Td("issue_type"),
                    html.Td("precision_0"),
                    html.Td("recall_0"),
                    html.Td("f1 score_0"),
                    html.Td("total number of 0"),

                    html.Td("precision_1"),
                    html.Td("recall_1"),
                    html.Td("f1 score_1"),
                    html.Td("total number of 1")
                ])] + [
                html.Tr([
                    html.Td(name),
                    html.Td(i[0]),
                    html.Td(j[0]),
                    html.Td(k[0]),
                    html.Td(s[0]),

                    html.Td(i[1]),
                    html.Td(j[1]),
                    html.Td(k[1]),
                    html.Td(s[1])
                ])
                for name, (i, j, k, s) in backend.train_model(model=model, do_cv= "cv" in cv )
            ])

        except AssertionError:
            return html.Table([
              html.Tr([
                  html.Td("issue_type"),
                  html.Td("precision_0"),
                  html.Td("recall_0"),
                  html.Td("f1 score_0"),
                  html.Td("total number of 0"),

                  html.Td("precision_1"),
                  html.Td("recall_1"),
                  html.Td("f1 score_1"),
                  html.Td("total number of 1")
              ])] + [
              html.Tr([
                  html.Td(name),
                  html.Td(i[0]),
                  html.Td(j[0]),
                  html.Td(k[0]),
                  html.Td(s[0]),

                  html.Td(i[1]),
                  html.Td(j[1]),
                  html.Td(k[1]),
                  html.Td(s[1])
              ])
              for name, (i, j, k, s) in backend.retrain(model=model, do_cv= "cv" in cv )
                ])


if __name__ == '__main__':
    app.run_server()
