# curl http://127.0.0.1:5000/infer

# curl -X POST -H "Content-Type: application/json" -d '{"datasetName": "test",
# "targetVariable": "var"}' http://127.0.0.1:5000/infer

# Previous version accepting csv files:
# curl http://127.0.0.1:5000/infer -F file=@assets/data/raw/cnc_without_target/02.csv
# curl http://127.0.0.1:5000/infer --data-binary @02.csv -H "Content-Type:text/plain"

# Testing API with JSON file as input
cat assets/data/raw/api_test/test_min.json | curl -X POST -H "Content-Type: application/json" -d @- http://127.0.0.1:5000/infer
