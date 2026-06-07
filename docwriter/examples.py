import generate_docs

# generate_docs.main(source_dir="D://REPOS//LLMlight//LLMlight", output_dir="D://temp//docs")
generate_docs.main(source_dir="D://REPOS//intruderscan//", output_dir="D://temp//docs/intruderscan/new")


# %%

# First run: discover + generate all pages
# python generate_docs.py --source-dir path/to/mylib --output-dir docs/

# # Fast rerun: skip discovery (manifest cached), regenerate RST only
# python generate_docs.py --source-dir path/to/mylib --output-dir docs/

# # Just discover, inspect, hand-tune rst_pages.json, then generate
# python generate_docs.py --source-dir path/to/mylib --discover-only
# # edit rst_pages.json ...
# python generate_docs.py --source-dir path/to/mylib --pages-file rst_pages.json
