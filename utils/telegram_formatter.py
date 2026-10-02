import re

def format_markdown_table(table_text: str) -> str:
    """Format markdown table into a padded ASCII table."""
    lines = [line.strip() for line in table_text.strip().split('\n') if line.strip()]
    if len(lines) < 2:
        return table_text
    
    # Parse rows
    raw_rows = []
    for line in lines:
        cells = [cell.strip() for cell in line.split('|')]
        # Strip outer empty cells from splitting leading/trailing pipes
        if line.startswith('|'):
            cells = cells[1:]
        if line.endswith('|') and len(cells) > 0:
            cells = cells[:-1]
        raw_rows.append(cells)
        
    headers = raw_rows[0]
    data_rows = raw_rows[2:]
    
    num_cols = max(len(headers), max((len(r) for r in data_rows), default=0))
    col_widths = [0] * num_cols
    
    def get_cell(row, idx):
        return row[idx] if idx < len(row) else ""
        
    for i in range(num_cols):
        w = len(get_cell(headers, i))
        for r in data_rows:
            w = max(w, len(get_cell(r, i)))
        col_widths[i] = max(w, 1)
        
    formatted_lines = []
    
    # Header row
    header_cells = [get_cell(headers, i).ljust(col_widths[i]) for i in range(num_cols)]
    formatted_lines.append("| " + " | ".join(header_cells) + " |")
    
    # Separator row
    sep_cells = ["-" * col_widths[i] for i in range(num_cols)]
    formatted_lines.append("| " + "-+-".join(sep_cells) + " |")
    
    # Data rows
    for r in data_rows:
        row_cells = [get_cell(r, i).ljust(col_widths[i]) for i in range(num_cols)]
        formatted_lines.append("| " + " | ".join(row_cells) + " |")
        
    return "\n".join(formatted_lines)

def format_for_telegram(text: str) -> str:
    """Convert Markdown to Telegram-safe HTML, keeping code/tables protected."""
    block_placeholders = []
    
    # Helper to protect code blocks
    def save_code_block(match):
        code = match.group(2)
        escaped_code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html = f"<pre><code>{escaped_code.strip()}</code></pre>"
        placeholder = f"BLOCKPLACEHOLDER{len(block_placeholders)}"
        block_placeholders.append(html)
        return placeholder

    # 1. Protect fenced code blocks (do this first)
    text = re.sub(r'```([a-zA-Z0-9_\-+]*)\n(.*?)\n```', save_code_block, text, flags=re.DOTALL)
    text = re.sub(r'```(.*?)\n```', save_code_block, text, flags=re.DOTALL)
    
    # 2. Protect markdown tables using strict validation
    lines = text.split('\n')
    i = 0
    separator_pat = re.compile(r'^\|?\s*:?-+:?\s*(?:\|\s*:?-+:?\s*)*\|?$')
    
    while i < len(lines):
        line = lines[i].strip()
        if '|' in line:
            # Check if next line is a separator row
            if i + 1 < len(lines) and separator_pat.match(lines[i+1].strip()):
                table_lines = [lines[i], lines[i+1]]
                i += 2
                while i < len(lines) and '|' in lines[i]:
                    table_lines.append(lines[i])
                    i += 1
                
                table_text = "\n".join(table_lines)
                formatted_table = format_markdown_table(table_text)
                escaped_table = formatted_table.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
                html = f"<pre><code>{escaped_table}</code></pre>"
                
                placeholder = f"BLOCKPLACEHOLDER{len(block_placeholders)}"
                block_placeholders.append(html)
                
                lines[i - len(table_lines)] = placeholder
                for _ in range(len(table_lines) - 1):
                    lines.pop(i - len(table_lines) + 1)
                i -= (len(table_lines) - 1)
            else:
                i += 1
        else:
            i += 1
            
    text = "\n".join(lines)
    
    # 3. Protect inline code
    inline_placeholders = []
    def save_inline_code(match):
        code = match.group(1)
        escaped_code = code.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
        html = f"<code>{escaped_code}</code>"
        placeholder = f"INLINEPLACEHOLDER{len(inline_placeholders)}"
        inline_placeholders.append(html)
        return placeholder
        
    text = re.sub(r'`([^`\n]+)`', save_inline_code, text)
    
    # 4. Escape remaining raw prose (ampersand first, then angle brackets)
    text = text.replace('&', '&amp;').replace('<', '&lt;').replace('>', '&gt;')
    
    # 5. Convert Markdown links: [anchor text](https://url)
    text = re.sub(
        r'\[([^\]\n]+)\]\(((?:https?://|tg://)[^\)\s]+)\)',
        r'<a href="\2">\1</a>',
        text
    )
    
    # 6. Apply bold and italic conversions (doubles first, then singles)
    text = re.sub(r'\*\*(.*?)\*\*', r'<b>\1</b>', text)
    text = re.sub(r'__(.*?)__', r'<b>\1</b>', text)
    text = re.sub(r'\*(.*?)\*', r'<i>\1</i>', text)
    text = re.sub(r'_(.*?)_', r'<i>\1</i>', text)
    
    # 6. Re-insert inline code placeholders
    for idx, html in enumerate(inline_placeholders):
        text = text.replace(f"INLINEPLACEHOLDER{idx}", html)
        
    # 7. Re-insert block placeholders
    for idx, html in enumerate(block_placeholders):
        text = text.replace(f"BLOCKPLACEHOLDER{idx}", html)
        
    return text

def split_message(html_text: str, max_chars: int = 4000) -> list[str]:
    """Split HTML text into Telegram-safe chunks of max_chars, keeping pre blocks intact."""
    pattern = re.compile(r'(<pre><code>.*?</code></pre>)', re.DOTALL)
    parts = pattern.split(html_text)
    
    chunks = []
    current_chunk = []
    current_len = 0
    
    for part in parts:
        if not part:
            continue
        
        if part.startswith('<pre><code>'):
            # If a single pre block exceeds the maximum length
            if len(part) > max_chars:
                if current_chunk:
                    chunks.append("".join(current_chunk))
                    current_chunk = []
                    current_len = 0
                
                # Split the block content internally by lines
                inner_content = part[len('<pre><code>'):-len('</code></pre>')]
                lines = inner_content.split('\n')
                
                sub_chunk = []
                sub_len = len('<pre><code></code></pre>')
                
                for line in lines:
                    line_str = line + '\n'
                    if sub_len + len(line_str) > max_chars:
                        chunks.append(f"<pre><code>{''.join(sub_chunk).rstrip()}</code></pre>")
                        sub_chunk = [line_str]
                        sub_len = len('<pre><code></code></pre>') + len(line_str)
                    else:
                        sub_chunk.append(line_str)
                        sub_len += len(line_str)
                        
                if sub_chunk:
                    chunks.append(f"<pre><code>{''.join(sub_chunk).rstrip()}</code></pre>")
            else:
                if current_len + len(part) > max_chars:
                    chunks.append("".join(current_chunk))
                    current_chunk = [part]
                    current_len = len(part)
                else:
                    current_chunk.append(part)
                    current_len += len(part)
        else:
            paragraphs = part.split('\n\n')
            for i, p in enumerate(paragraphs):
                p_str = p if i == 0 else '\n\n' + p
                
                if len(p_str) > max_chars:
                    if current_chunk:
                        chunks.append("".join(current_chunk))
                        current_chunk = []
                        current_len = 0
                        
                    lines = p_str.split('\n')
                    for line in lines:
                        line_str = line + '\n'
                        if len(line_str) > max_chars:
                            for k in range(0, len(line_str), max_chars):
                                chunks.append(line_str[k:k+max_chars])
                        else:
                            if current_len + len(line_str) > max_chars:
                                chunks.append("".join(current_chunk))
                                current_chunk = [line_str]
                                current_len = len(line_str)
                            else:
                                current_chunk.append(line_str)
                                current_len += len(line_str)
                else:
                    if current_len + len(p_str) > max_chars:
                        chunks.append("".join(current_chunk))
                        current_chunk = [p_str.lstrip('\n')]
                        current_len = len(current_chunk[0])
                    else:
                        current_chunk.append(p_str)
                        current_len += len(p_str)
                        
    if current_chunk:
        chunks.append("".join(current_chunk))
        
    return [c.strip() for c in chunks if c.strip()]
