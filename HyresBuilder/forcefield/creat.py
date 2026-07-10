import itertools

# 1. Base list of known atom types (from your previous bonds list)
base_types = [
    "C2E", "C3E", "CT", "A2V", "A1L", "A1I", "A5M", "P5N", "QaD", "P4Q",
    "QaE", "P1C", "P1S", "P1T", "A2P", "A3K", "QdK", "A3R", "QdR", "A4H",
    "P1H", "P2H", "A1F", "A2F", "A3F", "A1Y", "A2Y", "P1Y", "A1W", "P1W",
    "A2W", "A3W", "A4W", "RP", "RS1", "RS2", "RA1", "RG1", "RA2", "RG2",
    "RA3", "RG4", "RC2", "RA4", "RG3", "RU2", "RC1", "RU1", "RC3", "RU3",
    "MG", "CAL", "PHO", "SMG", "M01", "M02", "M03", "M04", "M05", "M06",
    "M07", "M08", "MCI", "MSO", "MSS", "MCL", "MCF", "MBR",
    "H", "NH1", "N", "C", "O", "P2S", "P3S", "P2T", "P3T", "PEG"
]

# 2. Your provided list of actual angles to skip
raw_angles_skip_list = """
A1I   M07   A2V      5.0     148.0
A2V   M02   M07      5.0     116.5
A2V   P1S   M03      5.0      77.0
A2V   P1T   M03      5.0     114.0
A2V   QdK   M02     25.0      85.0
A2V   QdK   C3E     10.0     133.0
A2W   A2W   M04      0.0     165.0
C3E   M07   A2V      5.0     115.0
M01   C3E   M02     35.0      62.0
M01   PHO   PHO     60.0     158.0
M01   PHO   RS1      5.0     168.0
M01   P1S   M02     35.0      91.0
M01   RS1   M05      5.0     150.0
M02   A2V   M02     10.0     134.8
M02   A2V   M06      5.0     122.0
M02   A2V   QdK      5.0     130.0
M02   C3E   C3E     30.0     105.6
M02   C3E   M02     15.0     170.0
M02   M03   P1C     10.0      82.7
M02   M06   M03      5.0     115.0
M02   P1S   C3E     25.0      86.0
M02   P1T   C3E     15.0     150.0
M02   P1T   M02     15.0      85.0
M02   QdK   P1C      5.0     102.0
M03   A2V   M02      5.0      82.8
M03   C3E   M02      5.0      90.0
M03   M03   M02     10.0     126.5
M03   M03   P1C     25.0     129.5
M05   RS2   M06     55.0     126.0
M05   RS2   P1H     55.0     126.0
M06   A2W   M04     15.0     166.7
M06   A2V   M07      5.0      96.0
M06   M03   M03     10.0     132.0
M06   M03   P1C     10.0     129.0
M06   P1T   M02      0.5     113.5
M06   QdK   M02      5.0     105.0
M06   RS1   M05      5.0     153.5
M06   RS1   RS2      5.0     142.0
M07   A2V   M02      5.0     132.0
A2V   M07   A2V      5.0     148.0
PHO   A2V   P1T     25.0      83.0
PHO   PHO   A2V     25.0     122.5
PHO   PHO   P1T      5.0     110.0
PHO   PHO   RS1     10.0     132.0
PHO   P1T   P1S      5.0      90.0
PHO   RS1   M02      5.0     126.0
PHO   RS1   M03      5.0      85.0
PHO   RS1   M05     20.0     105.5
PHO   RS1   P1S     30.0      97.0
PHO   RS1   RS2     30.0      97.0
P1C   RS1   M05      5.0     137.5
P1C   RS1   RS2      5.0     142.0
P1H   A2W   M04     15.0     166.7
P1S   A2V   M01      5.0     104.0
P1S   A2V   P1S      5.0      73.0
P1S   C3E   M02     15.0     161.0
P1S   M03   QaD      5.0     131.0
P1S   P1T   P1H      5.0     145.0
P1S   P1T   RP       5.0     115.0
P1S   RS1   M05      5.0     137.5
P1S   RS1   RS2      5.0     142.0
P1T   C3E   M02     60.0      57.5
P1T   M03   M03     10.0     136.5
P1T   P1H   A1W     25.0      85.0
P1T   P1S   P1T      5.0      90.0
P1T   RP    C3E      5.0     103.5
QdK   C3E   M02      0.5     106.5
QdK   M06   RS1      5.0     145.0
QdK   P1C   QdK      5.0     121.5
RA1   RS2   M05     15.0     144.0
RC1   RS2   M05      5.0     132.0
RG1   RS2   M05      5.0     132.0
RP    C3E   M06      5.0      95.5
RS1   M05   M01      5.0     141.5
RS1   RS2   M06     55.0     144.0
RS1   RS2   P1H     55.0     144.0
RS2   M05   M01      5.0      81.5
RS2   M06   A2W     50.0     149.0
RS2   P1H   A2W     50.0     149.0
RU1   RS2   M05      5.0     132.0
C3E   C1E   NH1   90.0     109.5
A2V   C1E   NH1   20.0     111.3
A1L   C1E   NH1   20.0     114.3
A1I   C1E   NH1   20.0     110.4
A5M   C1E   NH1   20.0     112.7
P5N   C1E   NH1   20.0     104.6
QaD   C1E   NH1   20.0     106.4
P4Q   C1E   NH1   20.0     111.0
QaE   C1E   NH1   20.0     113.7
P1C   C1E   NH1   20.0     107.0
P1S   C1E   NH1   20.0     114.2
P1T   C1E   NH1   20.0     106.9
A3K   C1E   NH1   20.0     112.3
A3R   C1E   NH1   20.0     112.2
A4H   C1E   NH1   20.0     112.8
A1F   C1E   NH1   20.0     108.2
A1Y   C1E   NH1   20.0     108.7
A1W   C1E   NH1   20.0     110.8
P2S   C1E   NH1   20.0     110.3
P2T   C1E   NH1   20.0      98.4
C1E   A3K   QdK   20.0     144.9
C1E   A3R   QdR   20.0     137.9
C1E   A4H   P1H   20.0     129.5
C1E   A4H   P2H   20.0     127.6
C1E   A1F   A2F   10.0     116.0
C1E   A1F   A3F   20.0     140.2
C1E   A1Y   A2Y   10.0     115.8
C1E   A1Y   P1Y   20.0     139.2
C1E   A1W   P1W   20.0     125.6
C1E   A1W   A2W   20.0     133.7
C1E   P2S   P3S   20.0     132.8
C1E   P2T   P3T   20.0      72.6
P1W   A2W   A3W   50.0     127.9
A1W   A2W   A4W   50.0      98.6 
C     N     A2P   0.00     161.9
P1H   A4H   P2H   0.00      60.0 
A4H   P2H   P1H   0.00      60.0 
P2H   P1H   A4H   0.00      60.0 
A1F   A2F   A3F   0.00      60.0 
A2F   A3F   A1F   0.00      60.0 
A3F   A1F   A2F   0.00      60.0 
A1Y   A2Y   P1Y   0.00      60.0 
A2Y   P1Y   A1Y   0.00      60.0 
P1Y   A1Y   A2Y   0.00      60.0 
A1W   P1W   A2W   0.00      60.0 
A1W   A2W   P1W   0.00      60.0 
A2W   A1W   P1W   0.00      60.0 
A2W   A3W   A4W   0.00      60.0 
A2W   A4W   A3W   0.00      60.0 
A3W   A2W   A4W   0.00      60.0 
N     C1E   A2P   0.00      60.0 
N     A2P   C1E   0.00      60.0 
C1E   N     A2P   0.00      60.0 
PEG   PEG   PEG   10.0     123.0 
RS1   RP    RS1    4.0     105.0
RP    RS1   RP     8.0      95.0
RP    RS1   RS2    8.0     112.0
RS1   RS2   RP    20.0      55.0
RS1   RS2   RA1    8.0     136.0
RS1   RS2   RG1    8.0     136.0
RS1   RS2   RC1   10.0     115.0
RS1   RS2   RU1   10.0     115.0
RS2   RA1   RA2   50.0     127.0
RS2   RG1   RG2   50.0     127.0
RS2   RA1   RA4   50.0     117.0
RS2   RG1   RG4   40.0     116.0
RS2   RC1   RC3  100.0      59.0
RS2   RU1   RU3  100.0      59.0
RA1   RA2   RA3  160.0      88.0
RA2   RA3   RA4  100.0      72.0
RA3   RA4   RA1  100.0      84.0
RG1   RG2   RG3  160.0      88.0
RG2   RG3   RG4  100.0      83.0
RG3   RG4   RG1  100.0      72.0
PHO   PHO   PHO  120.0      99.0
"""

# 3. Helper function to standardize angles (A-B-C is the same as C-B-A)
def standardize_angle(a1, a2, a3):
    # The central atom (a2) stays in the middle. 
    # We sort the outer atoms alphabetically to easily catch duplicates.
    outer = sorted([a1, a3])
    return (outer[0], a2, outer[1])

# 4. Parse the skip list, strip comments, and find all unique atom types
existing_angles = set()
all_types = set(base_types)

for line in raw_angles_skip_list.split("\n"):
    # Ignore comments and empty lines
    line = line.split('!')[0].strip()
    parts = line.split()
    
    if len(parts) >= 3:
        t1, t2, t3 = parts[0], parts[1], parts[2]
        
        # Add to our type pool in case there are new types hiding in the skip list
        all_types.update([t1, t2, t3])
        
        # Standardize and save the angle to our skip list
        angle = standardize_angle(t1, t2, t3)
        existing_angles.add(angle)

# Sort the final master list of atom types alphabetically
types = sorted(list(all_types))

# 5. Generate the trick angles
trick_angles_count = 0
with open("trick_angles_filtered.inp", "w") as f:
    # Iterate over all possible central atoms
    for central in types:
        # Iterate over all combinations (with replacement) for the two outer atoms
        for outer1, outer2 in itertools.combinations_with_replacement(types, 2):
            angle = standardize_angle(outer1, central, outer2)
            
            # Only write if it's NOT a real angle
            if angle not in existing_angles:
                f.write(f"{angle[0]:<5} {angle[1]:<5} {angle[2]:<5}   0.0    1.0\n")
                trick_angles_count += 1

print(f"Total unique atom types processed: {len(types)}")
print(f"Total actual angles skipped: {len(existing_angles)}")
print(f"Total trick angles successfully created: {trick_angles_count}")