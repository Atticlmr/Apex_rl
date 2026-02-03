# Configuration file for the Sphinx documentation builder.
#
# For the full list of built-in configuration values, see the documentation:
# https://www.sphinx-doc.org/en/master/usage/configuration.html

# -- Project information -----------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#project-information

project = "apex-rl"
copyright = "2026, Atticlmr"
author = "Atticlmr"
release = "0.0.1"

# -- General configuration ---------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#general-configuration

extensions = []

templates_path = ["_templates"]
exclude_patterns = []


# -- Options for HTML output -------------------------------------------------
# https://www.sphinx-doc.org/en/master/usage/configuration.html#options-for-html-output

html_theme = "furo"
html_static_path = ["_static"]


# Furo 专属优化配置（可选但推荐）
html_theme_options = {
    # 显示源码仓库链接（右上角）
    "source_repository": "https://github.com/Atticlmr/Apex-rl",
    "source_branch": "main",
    "source_directory": "docs/source/",
    # 导航配置
    "navigation_with_keys": True,  # 键盘方向键导航
    # 页面底部显示 "Edit on GitHub"
    "footer_icons": [
        {
            "name": "GitHub",
            "url": "https://github.com/Atticlmr/Apex-rl",
            "html": "",
            "class": "fa-brands fa-github",
        },
    ],
}


# 侧边栏配置（Furo 特色：左侧目录 + 右侧页面内导航）
html_sidebars = {
    "**": [
        "sidebar/scroll-start.html",
        "sidebar/brand.html",
        "sidebar/search.html",
        "sidebar/navigation.html",
        "sidebar/ethical-ads.html",  # 如果使用 ReadTheDocs 托管则保留
        "sidebar/scroll-end.html",
    ]
}
