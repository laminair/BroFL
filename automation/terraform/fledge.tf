# All resources provisioned here are Ubuntu 18.04 x86 machines.

terraform {
  required_providers {
    opennebula = {
      source  = "opennebula/opennebula"
      version = "1.1.1"
    }
  }
}

provider "opennebula" {
  username      = var.ON_USERNAME
  password      = var.ON_PASSWD
  endpoint      = "http://xml-rpc..."
  flow_endpoint = "http://oneflow..."
  insecure      = true
}

resource "opennebula_virtual_machine" "flbench_orchestrator" {
  count = 1
  name        = "flbench_orchestrator_${count.index}"
  description = "FLBench Orchestrator"
  cpu         = 2
  vcpu        = 2
  memory      = 6144 # 4GB + 2GB for hypervisor space reservation
  group       = "<YOUR_GROUP>"
  template_id = 110

  tags = {
    machine_count = count.index
  }

  disk {
    image_id = 41
    driver   = "qcow2"
    size     = 16384 # MB
    target   = "vda"
  }

  # THIS IS NEW. We use the password to attach our NAS.
  context = {
    RBG_PASSWORD = var.ON_PASSWD
  }
}

resource "opennebula_virtual_machine" "flbench_server" {
  count = 1
  name        = "flbench_server_${count.index}"
  description = "FLBench Server"
  cpu         = 8
  vcpu        = 8
  memory      = 33792 # 32GB + 2GB for hypervisor space reservation
  group       = "woi-research"
  template_id = 110

  tags = {
    machine_count = count.index
  }

  disk {
    image_id = 41
    driver   = "qcow2"
    size     = 30720 # MB
    target   = "vda"
  }

  # THIS IS NEW. We use the password to attach our NAS.
  context = {
    RBG_PASSWORD = var.ON_PASSWD
  }
}

resource "opennebula_virtual_machine" "flbench_client" {
  count = 5
  name        = "flbench_client_${count.index}"
  description = "FLBench Client"
  cpu         = 4
  vcpu        = 4
  memory      = 6144 # 4GB + 2GB for hypervisor space reservation
  group       = "woi-research"
  template_id = 110

  tags = {
    machine_count = count.index
  }

  disk {
    image_id = 41
    driver   = "qcow2"
    size     = 30720 # MB
    target   = "vda"
  }

  # THIS IS NEW. We use the password to attach our NAS.
  context = {
    RBG_PASSWORD = var.ON_PASSWD
  }
}

resource "local_file" "ansible_inventory" {
  content = templatefile("ansible_inventory.tmpl",
    {
      flbench_clients = opennebula_virtual_machine.flbench_client.*.ip,
      flbench_server = opennebula_virtual_machine.flbench_server.*.ip,
      flbench_orchestrator = opennebula_virtual_machine.flbench_orchestrator.*.ip
    }
  )
  filename = "../ansible/inventory.cfg"
}
