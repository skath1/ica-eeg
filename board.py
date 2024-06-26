from brainflow.board_shim import BoardShim, BoardIds, BrainFlowInputParams

class BoardManager:
    def __init__(self, dev=False):
        self.board = None
        self.dev = dev
        self.args = {
            "timeout": 0,
            "ip_port": 0,
            "ip_protocol": 0,
            "ip_address": "",
            "serial_port": "/dev/cu.usbserial-DP04VYKA",     # correct usb c port on collection laptop is /dev/cu.usbserial-DP04VYKA,
            "mac_address": "",
            "other_info": "",
            "serial_number": "",
            "board_id": BoardIds.CYTON_DAISY_BOARD,
            "file": "",
            "master_board": BoardIds.NO_BOARD
        }

    def setup_board(self):
        BoardShim.enable_dev_board_logger()
        params = self.create_params()
        if self.dev:
            self.args["board_id"] = BoardIds.SYNTHETIC_BOARD
        self.board = BoardShim(self.args["board_id"], params)
        self.board.prepare_session()
        self.board.start_stream()

    def create_params(self):
        params = BrainFlowInputParams()
        params.ip_port = self.args["ip_port"]
        params.serial_port = self.args["serial_port"]
        params.mac_address = self.args["mac_address"]
        params.other_info = self.args["other_info"]
        params.serial_number = self.args["serial_number"]
        params.ip_address = self.args["ip_address"]
        params.ip_protocol = self.args["ip_protocol"]
        params.timeout = self.args["timeout"]
        params.file = self.args["file"]
        params.master_board = self.args["master_board"]
        return params

    def stop_stream(self):
        if self.board:
            self.board.stop_stream()

    def release_session(self):
        if self.board:
            self.board.release_session()

    def get_board_data(self):
        if self.board:
            return self.board.get_board_data()
        return None

    def start_stream(self):
        if self.board:
            try:
                self.stop_stream()
                self.board.start_stream()
            except:
                self.board.start_stream()