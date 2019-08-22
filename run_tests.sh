#!/bin/bash

# simple tests of functionality

python runway_model.py &
SERVER_PID=$!

sleep 10 # given server time to start

echo -e "\n\nTesting classify..."
curl -X POST -H "Content-Type: application/json" -d @test_files/smile_crop.json http://0.0.0.0:8000/classify

echo -e "\n\nTesting detect (single)..."
curl -X POST -H "Content-Type: application/json" -d @test_files/smile.json http://0.0.0.0:8000/detect

echo -e "\n\nTesting detect (multi)..."
curl -X POST -H "Content-Type: application/json" -d @test_files/combined.json http://0.0.0.0:8000/detect

echo -e "\n\nTesting detect_and_classify (single)..."
curl -X POST -H "Content-Type: application/json" -d @test_files/smile.json http://0.0.0.0:8000/detect_and_classify

echo -e "\n\nTesting detect_and_classify (multi)..."
curl -X POST -H "Content-Type: application/json" -d @test_files/combined.json http://0.0.0.0:8000/detect_and_classify

kill $SERVER_PID
