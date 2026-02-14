import sys
import struct
import json
import os
from rich.console import Console
from rich.table import Table
from rich.panel import Panel
from rich.prompt import Prompt, IntPrompt
from nfc_prototype.core import NFCPrototype

console = Console()

class NFCExplorer:
    def __init__(self, nfc_path):
        self.nfc_path = nfc_path
        self.proto = NFCPrototype()
        self.index = self.proto.get_stream_index(nfc_path)
        self.blocks = []
        self._scan_blocks()

    def _scan_blocks(self):
        if not self.index:
            console.print("[red]File is not indexed or corrupted.[/red]")
            return

        if isinstance(self.index, dict):
            offsets = self.index.get("offsets", [])
            names = self.index.get("names", [f"Block {i}" for i in range(len(offsets))])
        else:
            offsets = self.index
            names = [f"Block {i}" for i in range(len(offsets))]

        with open(self.nfc_path, "rb") as f:
            for i, offset in enumerate(offsets):
                f.seek(offset)
                hdr = f.read(34)
                if len(hdr) < 34: break
                
                meta_len = struct.unpack('!Q', hdr[16:24])[0]
                payload_len = struct.unpack('!Q', hdr[24:32])[0]
                
                meta_json = f.read(meta_len)
                try:
                    meta = json.loads(meta_json)
                except:
                    meta = {"error": "Corrupt Metadata"}
                
                flags = hdr[5]
                is_dedup = (flags & self.proto.DEDUP_POINTER_FLAG) != 0

                self.blocks.append({
                    "id": i,
                    "name": names[i] if i < len(names) else f"Block {i}",
                    "offset": offset,
                    "compressed_size": payload_len,
                    "original_size": meta.get("orig_bytes", 0),
                    "dtype": meta.get("dtype", "unknown"),
                    "stats": meta.get("stats", None),
                    "is_dedup": is_dedup
                })

    def show_summary(self):
        table = Table(title=f"NFC Explorer: {os.path.basename(self.nfc_path)}")
        table.add_column("ID", justify="right", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        table.add_column("Dtype", style="green")
        table.add_column("Size (Comp)", justify="right")
        table.add_column("Ratio", justify="right")
        table.add_column("Dedup", justify="center")
        
        total_orig = 0
        total_comp = 0

        for b in self.blocks:
            orig = b["original_size"]
            comp = b["compressed_size"]
            total_orig += orig
            total_comp += comp
            
            ratio = f"{orig/comp:.2f}x" if comp > 0 else "N/A"
            dedup_mark = "Yes" if b["is_dedup"] else "-"
            
            table.add_row(str(b["id"]), b["name"], b["dtype"], f"{comp:,}", ratio, dedup_mark)

        console.print(table)
        
        overall_ratio = f"{total_orig/total_comp:.2f}x" if total_comp > 0 else "N/A"
        console.print(Panel(f"Total Blocks: {len(self.blocks)} | Original: {total_orig:,} bytes | Compressed: {total_comp:,} bytes | Ratio: {overall_ratio}", title="Summary", style="bold blue"))

    def inspect_block(self, block_id):
        if block_id < 0 or block_id >= len(self.blocks):
            console.print("[red]Invalid Block ID[/red]")
            return

        b = self.blocks[block_id]
        console.print(Panel(json.dumps(b, indent=4), title=f"Block {block_id}: {b['name']}", style="yellow"))

    def run(self):
        self.show_summary()
        while True:
            choice = Prompt.ask("[bold]Action[/bold] (inspect [ID] / quit)", default="quit")
            if choice.lower() in ["q", "quit", "exit"]:
                break
            elif choice.lower().startswith("inspect"):
                parts = choice.split()
                if len(parts) > 1 and parts[1].isdigit():
                    self.inspect_block(int(parts[1]))
                else:
                    console.print("[red]Usage: inspect <ID>[/red]")
            else:
                pass

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m nfc_prototype.explorer <file.nfc>")
        sys.exit(1)
    
    explorer = NFCExplorer(sys.argv[1])
    explorer.run()
