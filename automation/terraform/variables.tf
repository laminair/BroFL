variable "ON_USERNAME" {
  description = "OpenNebula Username"
  type        = string
  default     = ""
}

variable "ON_PASSWD" {
  description = "OpenNebula Username"
  type        = string
  default     = ""
  sensitive   = true
}
