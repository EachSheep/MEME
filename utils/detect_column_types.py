import re
from collections import Counter

def detect_column_types(table):
    """
    Detect column types in a markdown table using comprehensive regex patterns.
    Returns a dictionary with column names as keys and detected types as values.
    """
    if not table:
        return {}
    
    column_types = {}
    
    # Define comprehensive regex patterns for different data types
    patterns = {
        # Numbers
        'percentage': re.compile(r'^\s*[+-]?\d+([,\d]*)?(\.\d+)?\s*%\s*$|^\s*[+-]?\.\d+\s*%\s*$'),
        'currency': re.compile(r'^\s*[\$€£¥₹₽₩¢₦₡₪₨₫]\s*[+-]?\d+([,\d]*)?(\.\d+)?\s*$|^\s*[+-]?\d+([,\d]*)?(\.\d+)?\s*[\$€£¥₹₽₩¢₦₡₪₨₫]\s*$|^\s*(USD|EUR|GBP|JPY|CNY|CAD|AUD)\s*[+-]?\d+([,\d]*)?(\.\d+)?\s*$'),
        'scientific_notation': re.compile(r'^\s*[+-]?\d+(\.\d+)?[eE][+-]?\d+\s*$'),
        'fraction': re.compile(r'^\s*[+-]?\d+\s*/\s*\d+\s*$'),
        'range_number': re.compile(r'^\s*\d+([,\d]*)?(\.\d+)?\s*[-–—~]\s*\d+([,\d]*)?(\.\d+)?\s*$'),
        'number_with_units': re.compile(r'^\s*[+-]?\d+([,\d]*)?(\.\d+)?\s*(kg|g|lb|oz|km|m|cm|mm|ft|in|L|ml|sec|min|hr|°C|°F|K|Hz|kHz|MHz|GHz|V|W|kW|MW|GB|MB|KB|TB|PB)\s*$', re.IGNORECASE),
        'integer': re.compile(r'^\s*[+-]?\d{1,3}(,\d{3})*\s*$|^\s*[+-]?\d+\s*$'),
        'float': re.compile(r'^\s*[+-]?\d+\.\d+\s*$|^\s*[+-]?\d{1,3}(,\d{3})*\.\d+\s*$'),

        # ID
        'id': re.compile(r'.*id.*'),
        
        # Dates and Times
        'date_iso': re.compile(r'^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s*$'),
        'date_us': re.compile(r'^\s*\d{1,2}[-/]\d{1,2}[-/]\d{2,4}\s*$'),
        'date_text': re.compile(r'^\s*(Jan|Feb|Mar|Apr|May|Jun|Jul|Aug|Sep|Oct|Nov|Dec|January|February|March|April|May|June|July|August|September|October|November|December)\s+\d{1,2},?\s+\d{4}\s*$', re.IGNORECASE),
        'date_dot': re.compile(r'^\s*\d{1,2}\.\d{1,2}\.\d{2,4}\s*$'),
        'time': re.compile(r'^\s*\d{1,2}:\d{2}(:\d{2})?\s*(AM|PM|am|pm)?\s*$'),
        'datetime': re.compile(r'^\s*\d{4}[-/]\d{1,2}[-/]\d{1,2}\s+\d{1,2}:\d{2}(:\d{2})?\s*$'),
        'year': re.compile(r'^\s*(19|20)\d{2}\s*$'),
        'quarter': re.compile(r'^\s*Q[1-4]\s+\d{4}\s*$|^\s*\d{4}\s*Q[1-4]\s*$', re.IGNORECASE),
        
        # Text patterns
        'email': re.compile(r'^\s*[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}\s*$'),
        'url': re.compile(r'^\s*(https?|ftp)://[^\s/$.?#].[^\s]*\s*$|^\s*www\.[^\s/$.?#].[^\s]*\s*$', re.IGNORECASE),
        'phone': re.compile(r'^\s*(\+\d{1,3}[\s-]?)?\(?\d{3}\)?[\s-]?\d{3}[\s-]?\d{4}\s*$|^\s*\d{3}[-.\s]\d{3}[-.\s]\d{4}\s*$'),
        'ip_address': re.compile(r'^\s*\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3}\s*$'),
        'zip_code': re.compile(r'^\s*\d{5}(-\d{4})?\s*$'),
        'alphanumeric_id': re.compile(r'^\s*[A-Za-z0-9]{3,}\s*$'),
        'code': re.compile(r'^\s*[A-Z]{2,}[-_]?\d+\s*$|^\s*\d+[-_]?[A-Z]{2,}\s*$'),
        
        # Boolean and categorical
        'boolean': re.compile(r'^\s*(true|false|yes|no|y|n|1|0|on|off|enabled|disabled)\s*$', re.IGNORECASE),
        'rating': re.compile(r'^\s*[1-5]\s*stars?\s*$|^\s*[★☆]{1,5}\s*$|^\s*\d+(\.\d+)?\s*/\s*\d+\s*$', re.IGNORECASE),
        'grade': re.compile(r'^\s*[A-F][+-]?\s*$|^\s*(excellent|good|fair|poor|outstanding)\s*$', re.IGNORECASE),
        
        # Coordinates and special formats
        'coordinates': re.compile(r'^\s*[+-]?\d+(\.\d+)?[°]?\s*[NS]?\s*,?\s*[+-]?\d+(\.\d+)?[°]?\s*[EW]?\s*$', re.IGNORECASE),
        'hex_color': re.compile(r'^\s*#[0-9A-Fa-f]{6}\s*$|^\s*#[0-9A-Fa-f]{3}\s*$'),
        'isbn': re.compile(r'^\s*(ISBN[-\s]?)?(97[89][-\s]?)?\d{1,5}[-\s]?\d{1,7}[-\s]?\d{1,6}[-\s]?[\dX]\s*$', re.IGNORECASE),
        'version': re.compile(r'^\s*v?\d+(\.\d+)*(-\w+)?\s*$', re.IGNORECASE),
        
        # Special null/empty indicators
        'null_indicator': re.compile(r'^\s*(null|none|n/a|na|nil|empty|blank|missing|unknown|tbd|pending)\s*$', re.IGNORECASE)
    }
    
    for column in table[0].keys():
        type_counts = Counter()
        total_values = 0
        
        for row in table:
            value = str(row[column]).strip()
            
            if not value or value == '':
                continue
                
            total_values += 1
            
            # Check each pattern
            detected = False
            for type_name, pattern in patterns.items():
                if pattern.match(value):
                    type_counts[type_name] += 1
                    detected = True
                    break
            
            if not detected:
                # Additional checks for mixed content
                if re.search(r'\d', value) and re.search(r'[a-zA-Z]', value):
                    type_counts['mixed_alphanumeric'] += 1
                elif value.isupper():
                    type_counts['uppercase_text'] += 1
                elif value.islower():
                    type_counts['lowercase_text'] += 1
                elif value.istitle():
                    type_counts['title_case'] += 1
                else:
                    type_counts['text'] += 1
        
        # Determine the most common type for this column
        if type_counts and total_values > 0:
            most_common_type, count = type_counts.most_common(1)[0]
            confidence = count / total_values
            
            # Merge related types into broader categories
            if most_common_type in ['integer', 'float', 'scientific_notation', 'fraction', 'percentage', 'currency', 'range_number', 'number_with_units']:
                column_types[column] = 'number'
            elif most_common_type in ['date_iso', 'date_us', 'date_text', 'date_dot', 'time', 'datetime', 'year', 'quarter']:
                column_types[column] = 'datetime'
            elif most_common_type in ['email', 'url', 'phone', 'ip_address', 'zip_code', 'alphanumeric_id', 'code', 'coordinates', 'hex_color', 'isbn', 'version']:
                column_types[column] = 'identifier'
            elif most_common_type in ['boolean', 'rating', 'grade']:
                column_types[column] = 'categorical'
            elif most_common_type in ['uppercase_text', 'lowercase_text', 'title_case', 'mixed_alphanumeric', 'text']:
                column_types[column] = 'text'
            elif most_common_type == 'null_indicator':
                column_types[column] = 'null'
            else:
                column_types[column] = 'text'
        else:
            column_types[column] = 'empty'
    
    return column_types
