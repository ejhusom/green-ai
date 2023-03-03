curl http://127.0.0.1:5000/virtual_sensors

curl -X POST -H "Content-Type: application/json" -d '{"datasetName": "test",
"targetVariable": "var"}' http://127.0.0.1:5000/virtual_sensors

curl http://127.0.0.1:5000/virtual_sensors
