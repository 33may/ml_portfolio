COLOR_BULL = 'rgba(38,166,154,0.9)'  # #26a69a
COLOR_BEAR = 'rgba(239,83,80,0.9)'  # #ef5350

chartMultipaneOptions = [
                {
                    "width": 1200,
                    "height": 400,
                    "layout": {
                        "background": {
                            "type": "solid",
                            "color": 'transparent'
                        },
                        "textColor": "white"
                    },
                    "grid": {
                        "vertLines": {
                            "color": "rgba(197, 203, 206, 0.2)"
                        },
                        "horzLines": {
                            "color": "rgba(197, 203, 206, 0.2)"
                        }
                    },
                    "crosshair": {
                        "mode": 0
                    },
                    "priceScale": {
                        "borderColor": "rgba(197, 203, 206, 0.8)"
                    },
                    "timeScale": {
                        "borderColor": "rgba(197, 203, 206, 0.8)",
                        "barSpacing": 15
                    },
                },
                {
                    "width": 1200,
                    "height": 100,
                    "layout": {
                        "background": {
                            "type": 'solid',
                            "color": 'transparent'
                        },
                        "textColor": 'white',
                    },
                    "grid": {
                        "vertLines": {
                            "color": 'rgba(42, 46, 57, 0)',
                        },
                        "horzLines": {
                            "color": 'rgba(42, 46, 57, 0.2)',
                        }
                    },
                    "timeScale": {
                        "visible": False,
                    }
                },
                {
                    "width": 1200,
                    "height": 200,
                    "layout": {
                        "background": {
                            "type": "solid",
                            "color": 'transparent'
                        },
                        "textColor": "white"
                    },
                    "timeScale": {
                        "visible": False,
                    },
                    "grid": {
                        "vertLines": {
                            "color": "rgba(197, 203, 206, 0.2)"
                        },
                        "horzLines": {
                            "color": "rgba(197, 203, 206, 0.2)"
                        }
                    },
                }
            ]


def seriesCandlestickChart(candles):
    return [
        {
            "type": 'Candlestick',
            "data": candles,
            "options": {
                "upColor": COLOR_BULL,
                "downColor": COLOR_BEAR,
                "borderVisible": False,
                "wickUpColor": COLOR_BULL,
                "wickDownColor": COLOR_BEAR
            }
        }
    ]

def seriesLineChart(candles, line_color="#26a69a"):
    return [{
        "type": "Line",
        "data": candles,
        "options": {
            "lineWidth": 2,
            "lineColor": line_color,
        }
    }]


def seriesVolumeChart(volume):
    return [
        {
            "type": 'Histogram',
            "data": volume,
            "options": {
                "priceFormat": {
                    "type": 'volume',
                },
                "priceScaleId": ""  # set as an overlay setting,
            },
            "priceScale": {
                "scaleMargins": {
                    "top": 0,
                    "bottom": 0,
                },
                "alignLabels": False
            }
        }
    ]


def seriesMACDchart(macd_fast, macd_slow, macd_hist):
    return [
        {
            "type": 'Line',
            "data": macd_fast,
            "options": {
                "color": 'blue',
                "lineWidth": 2
            }
        },
        {
            "type": 'Line',
            "data": macd_slow,
            "options": {
                "color": 'green',
                "lineWidth": 2
            }
        },
        {
            "type": 'Histogram',
            "data": macd_hist,
            "options": {
                "color": 'red',
                "lineWidth": 1
            }
        }
    ]
