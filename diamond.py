# Function to read CSV file
def read_csv(filename):
    with open(filename, 'r') as file:
        data = file.readlines()
    # Extract headers
    headers = data[0].strip().split(',')
    # Extract rows
    rows = [line.strip().split(',') for line in data[1:]]
    return headers, rows

# Load dataset
headers, rows = read_csv('dimond.csv')

# Display the first few rows
for row in rows[:5]:
    print(row)

# Function to convert strings to floats
def to_float(value):
    try:
        return float(value)
    except ValueError:
        return None

# Basic statistics
def basic_statistics(data):
    prices = [to_float(row[1]) for row in data]  # Assuming price is in the second column
    valid_prices = [price for price in prices if price is not None]
    
    if valid_prices:
        avg_price = sum(valid_prices) / len(valid_prices)
        min_price = min(valid_prices)
        max_price = max(valid_prices)

        print(f'Average Price: {avg_price:.2f}')
        print(f'Min Price: {min_price}')
        print(f'Max Price: {max_price}')
    else:
        print("No valid prices found.")

# Call basic statistics function
basic_statistics(rows)

# Grouping data by cut
def group_by_cut(data):
    cut_prices = {}
    
    for row in data:
        cut = row[0]  # Assuming cut is in the first column
        price = to_float(row[1])  # Assuming price is in the second column
        
        if cut not in cut_prices:
            cut_prices[cut] = []
        if price is not None:
            cut_prices[cut].append(price)
    
    # Calculate average price by cut
    for cut, prices in cut_prices.items():
        avg_price = sum(prices) / len(prices) if prices else 0
        print(f'Average Price for {cut}: {avg_price:.2f}')

# Call grouping function
group_by_cut(rows)
