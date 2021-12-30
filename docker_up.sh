#!/bin/bash
docker build . -t datium_a:latest
docker run -ti -p 5000:5000 datium_a:latest
