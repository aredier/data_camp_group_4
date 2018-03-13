from datetime import date
import warnings

import dash
from dash.dependencies import Input, Output, State
import dash_core_components as dcc
import dash_html_components as html
import plotly.graph_objs as go
import arrow
import base64

from backend import ReviewApp

app = dash.Dash()
app.config['suppress_callback_exceptions'] = True

backend = ReviewApp("data/data.db")
warnings.filterwarnings("ignore")
#backend.build_data_base(labeled="data/labeled_data.db",unlabeled="data_unlabeled.csv", log_file="data/scraper_2.log")
backend._build_vocab(preprocess=True)


ISSUE_NAMES = backend.issue_categories

def compute_issue_phone_pie_chart():
    issue_count = [(k,v) for k, v in backend._base.get_phone_issue_count(True).items()]
    return {
        "data": [go.Pie(
            labels=[x[0] for x in issue_count],
            values=[x[1] for x in issue_count], 
            marker = dict(colors= ['#F3C2BC', '#F7C9B5', '#F2DBA0', '#EEEDA2', '#DCF6A4', '#9EF48F', '#90F2D2', '#84E8E9', '#84ACED','#C396E9','#EF99EB'], 
                           line=dict(color='#000000', width=1))
        )]
    }


# LAYOUT

colors = {
    'generalbackground' : '#F0F8FD',
    'background': '#D5ECF8',
    'text': '#0583C7'
}

image_filename = 'ressources/the_insighter.png'
encoded_image = base64.b64encode(open(image_filename, 'rb').read())

app.layout = html.Div(style={'backgroundColor': colors['generalbackground']}, children =
    [html.Div([html.Img(src='data:image/png;base64,{}'.format(encoded_image.decode()))]),
     html.Div(style={'backgroundColor': colors['background']}, children = 
            [html.H1(children="Detection of Smartphone issues", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '300%', 
            'font-family':'Impact'
        })]),
    html.Div(children = '''Welcome on your platform to visualize the main issues customer 
             encounter and complain about on smartphones. First, you will be able to choose the 
             way you train your model. Three options are available : XGBoost (which is recommanded 
             for better prediction but is computationally intensive), Random Forest and Logistic 
             Regression (the fastest computationally). Thanks to the training, you will be able 
             to set the prediction for each comment to be an issue, and which issue it is. Then, 
             you can visualize the proportion of each issues customer encounter and identify the 
             main sources of unsatisfaction. We provide also a plot showing the evolution of the issues over time.
             Finally, you can have a look at the comments containing an issue, selecting the issue 
             type you want to see, and eventually update the predictions if wrong to improve the model.''', 
             style ={'font_family' : 'Georgia',
                     'font-size': '120%'}),
    html.Div([
        html.H2(children="Training your model", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '200%', 
            'font-family':'Impact'
        }),
        html.Div(children= "Choose how you want to train your model:", 
             style ={'font_family' : 'Georgia',
                     'font-size': '120%'}),
        html.Div([
            dcc.Dropdown(
                id = "train_dopdown",
                options = [
                    {"label" : "XGBoost (recommended)", "value" : "xgb"},
                    {"label" : "Random Forest", "value" : "rf"},
                    {"label" : "Logistic Regression", "value" : "logreg"}
                ],
                value = "xgb"
            ),
            dcc.Checklist(
                options = [
                    {"label" : "Do cross validation" , "value" : "cv"}
                ],
                id = "do_cv",
                values = [], 
                style ={'font_family' : 'Georgia', 'font-size':'110%'}
            ),
            html.Button(children="Train", 
                        id="train_button", 
                        style ={'font_family' : 'Georgia', 
                                'font-size':'110%', 
                                'backgroundColor': colors['background'], 
                                'align-self': 'center'}),
            html.Div(children= '''Then, once you have trained your model, 
                     you can update the prediction of the issues.''', 
             style ={'font_family' : 'Georgia',
                     'font-size': '120%'}),
            html.Button("Update predictions (beware, very long the first time)", 
                        id="update_predictions_button", 
                        style ={'font_family' : 'Georgia', 
                                'font-size':'110%', 
                                'backgroundColor': colors['background']}),
            html.P(id="update_error_message"),
            html.Div(id="train_resume")
        ],
        style = {"display" : "inline"}
        ),
    ]),
    html.Div([
        html.H2(children="Visualisation of the results", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '200%', 
            'font-family':'Impact'
        }),
            html.Div(children= '''Here are the proportions of each issues customers complain about.
                     By default, you will only see the proportion of each issue we hand-labeled. 
                     By clicking on the button 'Include the prediction of the model', 
                     you will see all the issues including the ones predicted.
                     ''',
                     style ={'font_family' : 'Georgia',
                     'font-size': '120%'})]),
    dcc.Checklist(
        id = "issue_type_source",
        options = [
            {"label" : 'Include the prediction of the model' , "value" : "predicted"}
        ],
        values = [],
        style ={'font_family' : 'Georgia', 'font-size':'110%'}
    ),
    html.H4(children="Provenence of the issues of smartphones", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '120%', 
            'font-family':'Georgia'
        }),
    dcc.Graph(id="issue_type_graph"),
    html.H4(children = "Issue type detected over time", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '120%', 
            'font-family':'Georgia'
        }),
    html.Div(
        id = "reviews_over_time"),
    html.H4(children="Phones concerned by the issues", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '120%', 
            'font-family':'Georgia'
        }),
    html.Div(id="issue_phone_graph"),
    html.H2(children="Comments containing issues", style={
            'textAlign': 'center',
            'color': colors['text'], 
            'font-size': '200%', 
            'font-family':'Impact'
        }),
    html.Div(children= '''Here you can see the comments containing issues. You can select one or 
             more issue types you want to see in the table. On the right-hand side of the table, if you notice 
             the issue contained in the comment is ill-predicted, you can update by hand the data 
             table, which will improve your model and give you better results for the next learning
             phase. (Still building this functionnality)
                     ''',
                     style ={'font_family' : 'Georgia',
                     'font-size': '120%'}),
    html.P(id="change_issue_message",
           children="placeholder"),
    dcc.Dropdown(
            id='categories',
            options=[{'label': i, 'value': i} for i in ISSUE_NAMES],
            multi=True
    ),
    dcc.DatePickerRange(
        id = "issue_date_picker",
        start_date = date(2018,1,1),
        end_date=arrow.get().date()
    ),
    html.Button("update datebase", id="update_database", style ={'font_family' : 'Georgia',
                                'font-size':'110%', 
                                'backgroundColor': colors['background'], 
                                'align-self': 'center'}),
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
            return "You can only update the predictions when the model is trained"

@app.callback(
    Output("issue_phone_graph", "children"),
    [
        Input("update_predictions_button", "n_clicks")
    ]
)
def return_issue_per_phone_graph(bs):
    if bs is None:
        bs = 0
    if backend.predicted:
        return dcc.Graph(id="phone_issue_graph", figure=compute_issue_phone_pie_chart())

    else:
        return html.P("Sorry, this component can only be loaded once the predictions have been updated")

@app.callback(
    Output("reviews_over_time", "children"),
   [
       Input("update_predictions_button", "n_clicks"),
   ]
)
def compute_reviews_over_time(bs):
    if bs is None:
        bs = 0
    if backend.predicted:
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
                            fillcolor=['#F3C2BC', '#F7C9B5', '#F2DBA0', '#EEEDA2', '#DCF6A4', '#9EF48F', '#90F2D2', '#84E8E9', '#84ACED','#C396E9','#EF99EB'], 
                            text=[str(i) for i in data[col]["value"]],
                            hoverinfo="text"
                        )
                        for i, col in enumerate(ISSUE_NAMES)
        ]))
    else:
        return html.P("Sorry, this component can only be loaded once the predictions have been updated")

@app.callback(
    Output("issue_type_graph", "figure"),
    [Input("issue_type_source", "values")]
)
          
def compute_issue_type_pie_chart(options):
    issue_count =[ (k, v) for k, v in backend.issue_type_count("predicted" in options).items()]
    return {'data' : [go.Pie(
                    labels = [x[0] for x in issue_count],
                    values = [x[1] for x in issue_count], 
                    marker = dict(colors= ['#F3C2BC', '#F7C9B5', '#F2DBA0', '#EEEDA2', 
                                           '#DCF6A4', '#9EF48F', '#90F2D2', '#84E8E9', '#84ACED',
                                           '#C396E9','#EF99EB'], 
                           line=dict(color='#000000', width=1))
        )]
    }

def rebuild_issues(start_date, end_date):
    start_date = arrow.get(start_date)
    end_date = arrow.get(end_date)
    issue_df = backend.find_issues(start_date=start_date, end_date=end_date)
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
        Input("categories", "value"),
        Input("issue_date_picker", "start_date"),
        Input("issue_date_picker", "end_date")
    ]
)
def get_new_issues(categories, start_date, end_date):
    if categories is None or categories == list():
        categories = ["issue"]
    return html.Table([
                html.Tr([
                    html.Td("Date of the issue", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        "width" : "10%",
                        'background': '#D5ECF8', 
                        'textAlign': 'center', 
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Comment containing the issue", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        "width" : "50%", 'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Label of the issue (if no label the model predicted there is an issue but cannot define which one)", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        "width" : "40%", 'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    })
                ])] + [
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
                        "border-collapse": "collapse"
                    }),
            html.Td(dcc.Dropdown(
                        id='issue_{}'.format(issue_dic["id"]),
                        options=[{'label': i, 'value': i} for i in ISSUE_NAMES],
                        multi=True,
                        value = issue_dic["issues"]
                     ),
                    style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"
                    })
        ])

        for issue_dic
        in rebuild_issues(start_date, end_date)
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
                    html.Td("Issue type", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Precision for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Recall for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("F1 score for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Total number of 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),

                    html.Td("Precision for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Recall for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("F1 score for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Total number of 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    })
                ])] + [
                html.Tr([
                    html.Td(name, style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(i[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(j[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(k[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(s[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),

                    html.Td(round(i[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(j[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(k[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(s[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"})
                ])
                for name, (i, j, k, s) in backend.retrain(model=model, do_cv= "cv" in cv )
            ])

        except AssertionError:
            return html.Table([
              html.Tr([
                    html.Td("Issue type", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Precision for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Recall for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("F1 score for 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Total number of 0", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),

                    html.Td("Precision for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Recall for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("F1 score for 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    }),
                    html.Td("Total number of 1", style={
                        "border": "1px solid black",
                        "border-collapse": "collapse",
                        'background': '#D5ECF8', 'textAlign': 'center',
                        'color': colors['text'], 'font-family':'Georgia'
                    })
              ])] + [
              html.Tr([
                    html.Td(name),
                    html.Td(round(i[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(j[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(k[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(s[0], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),

                    html.Td(round(i[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(j[1], 2), style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(k[1], 2),  style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"}),
                    html.Td(round(s[1], 2),  style={
                        "border": "1px solid black",
                        "border-collapse": "collapse"})
              ])
              for name, (i, j, k, s) in backend.retrain(model=model, do_cv= "cv" in cv )
                ])


if __name__ == '__main__':
    app.run_server()
