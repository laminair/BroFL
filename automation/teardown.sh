#!/usr/bin/env bash

cd $( dirname -- "$( readlink -f -- "$0"; )"; ) # Here we make sure to set the right pwd
cd terraform

# We remove all virtualized resources. No need to undeploy with Ansible.
terraform destroy -var-file="credentials.tfvars"

# We undeploy the project from all bare-metal infrastructure.
ansible-playbook -c ssh -i ansible/inventory_static.cfg ansible/teardown-project.yaml
