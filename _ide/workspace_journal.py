# 2026-09-03T11:05:29.446689400
import vitis

client = vitis.create_client()
client.set_workspace(path="CG4002_HLS")

comp = client.create_hls_component(name = "CG4002_HLS",part = "xczu3eg-sbva484-1-e",cfg_file = ["hls_config.cfg"],template = "empty_hls_component")

