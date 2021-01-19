# Requirements

- Python 3.8 and above
- A mapbox account (it's free).
- A heroku account if you want to deploy to heroku.

# How to deploy locally?

1. Create a file called `.env` and put your MAPBOX token there.

```
MAPBOX=<insert mapbox token here>
```

2. Create your virtual env with `python3 -m venv venv`
3. Activate the virtual env with `source venv/bin/activate`
4. Install the required packages / modules with `pip install -r requirements.txt`
5. Run `python3 app.py`


# How to deploy to heroku?

Read this [deployment guide](https://dash.plotly.com/deployment) by Plot.ly.