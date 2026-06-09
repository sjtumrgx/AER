#!/bin/bash
sudo docker stop foxy_controller || true
sudo docker rm foxy_controller || true
cd ~/gym/gym_deploy/docker/
sudo make autostart