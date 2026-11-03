"""SNOMED-CT diagnosis vocabulary, official PhysioNet-2021 equivalences, and clinical text anchors.

Descriptions paraphrase standard ECG interpretation criteria (AHA/ACCF/HRS 2009 recommendations
for the standardization and interpretation of the ECG). They are written once, before any training,
and are frozen for all experiments.
"""

# PhysioNet/CinC 2021 scored codes that the challenge treats as the same class.
EQUIVALENT = {
    "733534002": "164909002",  # complete LBBB -> LBBB
    "713427006": "59118001",   # complete RBBB -> RBBB
    "284470004": "63593006",   # PAC -> SVPB
    "427172004": "17338001",   # PVC -> VPB
}

# code: (abbreviation, name, description)
LABELS = {
    "164889003": ("AF", "atrial fibrillation",
        "Irregularly irregular RR intervals with no distinct P waves; the baseline shows fibrillatory "
        "f waves and the ventricular rhythm varies beat to beat."),
    "164890007": ("AFL", "atrial flutter",
        "Regular atrial activity at about 250 to 350 per minute producing sawtooth flutter waves, best "
        "seen in leads II, III, aVF and V1, often with fixed 2:1 or variable atrioventricular conduction."),
    "6374002": ("BBB", "bundle branch block",
        "Delayed ventricular conduction through one bundle branch giving a widened QRS complex of 120 ms "
        "or more with abnormal terminal QRS morphology."),
    "426627000": ("Brady", "bradycardia",
        "Ventricular rate below 60 beats per minute regardless of the underlying rhythm."),
    "164909002": ("LBBB", "left bundle branch block",
        "QRS duration of 120 ms or more with broad notched R waves in I, aVL, V5 and V6, absent septal q "
        "waves in lateral leads and deep S waves in V1, with discordant ST-T changes."),
    "59118001": ("RBBB", "right bundle branch block",
        "QRS duration of 120 ms or more with an rsR' pattern in V1 and V2 and a wide slurred S wave in "
        "leads I and V6."),
    "270492004": ("IAVB", "first degree atrioventricular block",
        "Every P wave is conducted to the ventricles but the PR interval is prolonged beyond 200 ms."),
    "713426002": ("IRBBB", "incomplete right bundle branch block",
        "rsR' pattern in V1 and V2 resembling right bundle branch block but with QRS duration between "
        "110 and 120 ms."),
    "39732003": ("LAD", "left axis deviation",
        "Mean frontal plane QRS axis more negative than minus 30 degrees: positive QRS in lead I and "
        "predominantly negative QRS in leads II and aVF."),
    "445118002": ("LAnFB", "left anterior fascicular block",
        "Left axis deviation between minus 45 and minus 90 degrees with qR in I and aVL, rS in II, III "
        "and aVF, and QRS duration under 120 ms."),
    "251146004": ("LQRSV", "low QRS voltages",
        "Small QRS amplitudes: peak-to-peak below 0.5 mV in all limb leads or below 1.0 mV in all "
        "precordial leads."),
    "698252002": ("NSIVCB", "nonspecific intraventricular conduction disturbance",
        "QRS duration above 110 ms without the morphology criteria of left or right bundle branch block."),
    "426783006": ("NSR", "sinus rhythm",
        "Normal sinus rhythm: upright P wave in leads I and II before every QRS complex, constant PR "
        "interval and a regular rate between 60 and 100 beats per minute."),
    "63593006": ("SVPB", "supraventricular premature beats",
        "Premature atrial or junctional beats: early narrow QRS complexes preceded by an abnormal or "
        "hidden P wave, usually followed by a non-compensatory pause."),
    "10370003": ("PR", "pacing rhythm",
        "Electronic pacemaker rhythm with sharp pacing spikes preceding paced P waves or wide paced QRS "
        "complexes."),
    "365413008": ("PRWP", "poor R wave progression",
        "R wave amplitude fails to increase normally across the precordial leads, remaining 3 mm or less "
        "in V3, with delayed transition zone."),
    "17338001": ("VPB", "ventricular premature beats",
        "Premature wide and bizarre QRS complexes not preceded by a P wave, with discordant T waves and "
        "usually a full compensatory pause."),
    "164947007": ("LPR", "prolonged PR interval",
        "PR interval longer than 200 ms, measured from the onset of the P wave to the onset of the QRS "
        "complex."),
    "111975006": ("LQT", "prolonged QT interval",
        "Heart-rate corrected QT interval prolonged, above about 450 ms in men or 460 ms in women, "
        "reflecting delayed ventricular repolarization."),
    "164917005": ("QAb", "Q wave abnormal",
        "Pathological Q waves wider than 40 ms or deeper than one quarter of the following R wave, "
        "suggesting prior myocardial necrosis."),
    "47665007": ("RAD", "right axis deviation",
        "Mean frontal plane QRS axis greater than plus 90 degrees: negative QRS in lead I and positive "
        "QRS in lead aVF."),
    "427393009": ("SA", "sinus arrhythmia",
        "Sinus P waves with normal morphology but phasic variation of the PP interval by more than 0.12 "
        "seconds, typically with respiration."),
    "426177001": ("SB", "sinus bradycardia",
        "Sinus rhythm with normal P waves before each QRS complex at a rate below 60 beats per minute."),
    "427084000": ("STach", "sinus tachycardia",
        "Sinus rhythm with normal P waves before each QRS complex at a rate above 100 beats per minute."),
    "164934002": ("TAb", "T wave abnormal",
        "Abnormal T waves that are flattened, low amplitude, biphasic or inverted in leads where they "
        "are normally upright, indicating abnormal repolarization."),
    "59931005": ("TInv", "T wave inversion",
        "Negative T waves in leads where the T wave is normally positive, such as I, II and V3 to V6."),
    # Frequent unscored codes, used as an extended vocabulary and for open-vocabulary tests.
    "164865005": ("MI", "myocardial infarction",
        "Evidence of myocardial infarction: pathological Q waves, loss of R wave amplitude and ST-T "
        "changes in contiguous leads matching a coronary territory."),
    "164873001": ("LVH", "left ventricular hypertrophy",
        "Increased QRS voltage over the left ventricle, for example S in V1 plus R in V5 or V6 above "
        "35 mm, often with lateral ST depression and T inversion of strain pattern."),
    "429622005": ("STD", "ST depression",
        "Horizontal or downsloping depression of the ST segment of at least 0.5 mm below the PR baseline "
        "in two contiguous leads."),
    "164931005": ("STE", "ST elevation",
        "Elevation of the ST segment at the J point above the baseline in two or more contiguous leads."),
    "55827005": ("LVHV", "left ventricular high voltage",
        "Increased QRS amplitudes in left precordial or limb leads meeting voltage criteria without "
        "other features of hypertrophy."),
    "195042002": ("IIAVB", "second degree atrioventricular block",
        "Intermittent failure of atrioventricular conduction: some P waves are not followed by a QRS "
        "complex, with or without progressive PR prolongation."),
    "233917008": ("AVB", "atrioventricular block",
        "Impaired conduction from atria to ventricles shown by prolonged PR interval or P waves not "
        "followed by QRS complexes."),
    "428750005": ("NSSTTA", "nonspecific ST-T abnormality",
        "Minor ST segment deviation or T wave flattening that does not meet criteria for ischemia, "
        "injury or a specific diagnosis."),
    "67741000119109": ("LAE", "left atrial enlargement",
        "Broad notched P wave longer than 120 ms in lead II with a deep terminal negative P component "
        "in V1."),
    "446358003": ("RAH", "right atrial hypertrophy",
        "Tall peaked P waves above 2.5 mm in the inferior leads indicating right atrial enlargement."),
    "89792004": ("RVH", "right ventricular hypertrophy",
        "Right axis deviation with dominant R wave in V1 and deep S waves in V5 and V6 indicating right "
        "ventricular hypertrophy."),
    "713422000": ("ATach", "atrial tachycardia",
        "Regular narrow complex tachycardia above 100 per minute with non-sinus P waves of uniform "
        "abnormal morphology."),
    "426761007": ("SVT", "supraventricular tachycardia",
        "Rapid regular narrow QRS tachycardia above 150 beats per minute originating above the "
        "ventricles, with P waves hidden or retrograde."),
    "251198002": ("CCR", "clockwise cardiac rotation",
        "Delayed precordial transition with the R/S ratio becoming greater than one only in V5 or V6."),
    "251199005": ("CCWR", "counterclockwise cardiac rotation",
        "Early precordial transition with dominant R waves already in V1 or V2."),
    "164912004": ("PWC", "P wave change",
        "Abnormal P wave morphology, amplitude or axis compared with normal sinus P waves."),
    "164942001": ("FQRS", "fragmented QRS",
        "Additional notches or RSR' patterns within a narrow QRS complex in two contiguous leads."),
    "164930006": ("STIAb", "ST interval abnormal",
        "Abnormal ST segment shape or level not meeting specific elevation or depression criteria."),
    "164937009": ("UAb", "U wave abnormal",
        "Prominent or inverted U waves following the T wave."),
}


def canonical(code: str) -> str:
    return EQUIVALENT.get(code, code)


SCORED_26 = [
    "164889003", "164890007", "6374002", "426627000", "164909002", "59118001", "270492004",
    "713426002", "39732003", "445118002", "251146004", "698252002", "426783006", "63593006",
    "10370003", "365413008", "17338001", "164947007", "111975006", "164917005", "47665007",
    "427393009", "426177001", "427084000", "164934002", "59931005",
]


def text_of(code: str, style: str = "full") -> str:
    """Text anchor for a code. style: 'name' (bare name) or 'full' (name + criteria) for ablation."""
    abbr, name, desc = LABELS[code]
    if style == "name":
        return f"An ECG showing {name}."
    return f"An ECG showing {name}. {desc}"
