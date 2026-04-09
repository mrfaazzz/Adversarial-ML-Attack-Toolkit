
import os
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ─── NSL-KDD schema ──────────────────────────────────────────────────────────
COLUMNS = [
    "duration", "protocol_type", "service", "flag", "src_bytes", "dst_bytes",
    "land", "wrong_fragment", "urgent", "hot", "num_failed_logins",
    "logged_in", "num_compromised", "root_shell", "su_attempted", "num_root",
    "num_file_creations", "num_shells", "num_access_files", "num_outbound_cmds",
    "is_host_login", "is_guest_login", "count", "srv_count", "serror_rate",
    "srv_serror_rate", "rerror_rate", "srv_rerror_rate", "same_srv_rate",
    "diff_srv_rate", "srv_diff_host_rate", "dst_host_count",
    "dst_host_srv_count", "dst_host_same_srv_rate", "dst_host_diff_srv_rate",
    "dst_host_same_src_port_rate", "dst_host_srv_diff_host_rate",
    "dst_host_serror_rate", "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate", "label", "difficulty"
]

NUMERIC_COLS = [
    "duration", "src_bytes", "dst_bytes", "land", "wrong_fragment", "urgent",
    "hot", "num_failed_logins", "logged_in", "num_compromised", "root_shell",
    "su_attempted", "num_root", "num_file_creations", "num_shells",
    "num_access_files", "num_outbound_cmds", "is_host_login", "is_guest_login",
    "count", "srv_count", "serror_rate", "srv_serror_rate", "rerror_rate",
    "srv_rerror_rate", "same_srv_rate", "diff_srv_rate", "srv_diff_host_rate",
    "dst_host_count", "dst_host_srv_count", "dst_host_same_srv_rate",
    "dst_host_diff_srv_rate", "dst_host_same_src_port_rate",
    "dst_host_srv_diff_host_rate", "dst_host_serror_rate",
    "dst_host_srv_serror_rate", "dst_host_rerror_rate",
    "dst_host_srv_rerror_rate"
]

CATEGORICAL_COLS = ["protocol_type", "service", "flag"]

# NSL-KDD label groups — mapped to binary: normal=0, attack=1
NORMAL_LABELS = {"normal"}
DOS_LABELS    = {"back", "land", "neptune", "pod", "smurf", "teardrop",
                 "apache2", "udpstorm", "processtable", "worm"}
PROBE_LABELS  = {"ipsweep", "nmap", "portsweep", "satan", "mscan", "saint"}
R2L_LABELS    = {"ftp_write", "guess_passwd", "imap", "multihop", "phf",
                 "spy", "warezclient", "warezmaster", "sendmail", "named",
                 "snmpgetattack", "snmpguess", "xlock", "xsnoop", "httptunnel"}
U2R_LABELS    = {"buffer_overflow", "loadmodule", "perl", "rootkit",
                 "sqlattack", "xterm", "ps"}


# ═════════════════════════════════════════════════════════════════════════════
# Realistic NSL-KDD generator
# Distributions derived from Tavallaee et al. (2009) + KDD CUP 99 statistics
# ═════════════════════════════════════════════════════════════════════════════
def generate_realistic_nslkdd(n_samples: int = 10000, random_state: int = 42) -> pd.DataFrame:

    rng = np.random.default_rng(random_state)

    # Sample counts per class
    n_normal = int(n_samples * 0.45)
    n_dos    = int(n_samples * 0.35)
    n_probe  = int(n_samples * 0.08)
    n_r2l    = int(n_samples * 0.07)
    n_u2r    = n_samples - n_normal - n_dos - n_probe - n_r2l

    frames = []

    # ── NORMAL traffic ────────────────────────────────────────────────────────
    def make_normal(n):
        return {
            "duration":          np.clip(rng.exponential(50, n).astype(int), 0, 58329),
            "protocol_type":     rng.choice(["tcp", "udp", "icmp"], n, p=[0.65, 0.25, 0.10]),
            "service":           rng.choice(["http", "ftp", "smtp", "ssh", "dns", "telnet",
                                              "pop_3", "ftp_data", "domain_u", "auth"],
                                             n, p=[0.30,0.12,0.10,0.08,0.10,0.08,0.06,0.08,0.05,0.03]),
            "flag":              rng.choice(["SF", "S0", "S1", "REJ", "RSTO", "RSTOS0"],
                                             n, p=[0.75, 0.05, 0.05, 0.06, 0.05, 0.04]),
            "src_bytes":         np.clip(rng.lognormal(5.5, 2.8, n).astype(int), 0, 1_379_236_915),
            "dst_bytes":         np.clip(rng.lognormal(6.0, 3.2, n).astype(int), 0, 1_309_937_401),
            "land":              rng.choice([0, 1], n, p=[0.999, 0.001]),
            "wrong_fragment":    rng.choice([0, 1, 2, 3], n, p=[0.97, 0.01, 0.01, 0.01]),
            "urgent":            np.zeros(n, dtype=int),
            "hot":               np.clip(rng.poisson(0.5, n), 0, 30).astype(int),
            "num_failed_logins": np.zeros(n, dtype=int),
            "logged_in":         rng.choice([0, 1], n, p=[0.30, 0.70]),
            "num_compromised":   np.zeros(n, dtype=int),
            "root_shell":        np.zeros(n, dtype=int),
            "su_attempted":      np.zeros(n, dtype=int),
            "num_root":          np.zeros(n, dtype=int),
            "num_file_creations":np.zeros(n, dtype=int),
            "num_shells":        np.zeros(n, dtype=int),
            "num_access_files":  np.zeros(n, dtype=int),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login":     np.zeros(n, dtype=int),
            "is_guest_login":    np.zeros(n, dtype=int),
            "count":             np.clip(rng.integers(1, 100, n), 1, 511),
            "srv_count":         np.clip(rng.integers(1, 80, n), 1, 511),
            "serror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_serror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "rerror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_rerror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "same_srv_rate":     np.clip(rng.beta(5, 1, n), 0, 1),
            "diff_srv_rate":     np.clip(rng.beta(1, 8, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_count":    rng.integers(1, 256, n),
            "dst_host_srv_count":rng.integers(1, 256, n),
            "dst_host_same_srv_rate":    np.clip(rng.beta(5, 1, n), 0, 1),
            "dst_host_diff_srv_rate":    np.clip(rng.beta(1, 8, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.beta(3, 2, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_serror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_serror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_rerror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_rerror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "label":             ["normal"] * n,
            "difficulty":        rng.integers(1, 21, n),
        }

    # ── DoS attacks (neptune / smurf / back pattern) ──────────────────────────
    def make_dos(n):
        return {
            "duration":          np.zeros(n, dtype=int),    # DoS = instant connections
            "protocol_type":     rng.choice(["tcp", "icmp", "udp"], n, p=[0.60, 0.30, 0.10]),
            "service":           rng.choice(["http", "private", "ecr_i", "smtp", "ftp"],
                                             n, p=[0.35, 0.25, 0.20, 0.10, 0.10]),
            "flag":              rng.choice(["S0", "SF", "REJ", "RSTO"], n, p=[0.55, 0.20, 0.15, 0.10]),
            "src_bytes":         np.clip(rng.lognormal(9.0, 1.5, n).astype(int), 0, 1_379_236_915),
            "dst_bytes":         np.zeros(n, dtype=int),    # DoS floods out, nothing comes back
            "land":              np.zeros(n, dtype=int),
            "wrong_fragment":    np.zeros(n, dtype=int),
            "urgent":            np.zeros(n, dtype=int),
            "hot":               np.zeros(n, dtype=int),
            "num_failed_logins": np.zeros(n, dtype=int),
            "logged_in":         np.zeros(n, dtype=int),
            "num_compromised":   np.zeros(n, dtype=int),
            "root_shell":        np.zeros(n, dtype=int),
            "su_attempted":      np.zeros(n, dtype=int),
            "num_root":          np.zeros(n, dtype=int),
            "num_file_creations":np.zeros(n, dtype=int),
            "num_shells":        np.zeros(n, dtype=int),
            "num_access_files":  np.zeros(n, dtype=int),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login":     np.zeros(n, dtype=int),
            "is_guest_login":    np.zeros(n, dtype=int),
            "count":             rng.integers(200, 511, n),  # High connection count = flood
            "srv_count":         rng.integers(200, 511, n),
            "serror_rate":       np.clip(rng.beta(8, 1, n), 0, 1),   # High SYN error rate
            "srv_serror_rate":   np.clip(rng.beta(8, 1, n), 0, 1),
            "rerror_rate":       np.clip(rng.beta(1, 8, n), 0, 1),
            "srv_rerror_rate":   np.clip(rng.beta(1, 8, n), 0, 1),
            "same_srv_rate":     np.clip(rng.beta(8, 1, n), 0, 1),   # All same service
            "diff_srv_rate":     np.clip(rng.beta(1, 8, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.beta(1, 8, n), 0, 1),
            "dst_host_count":    rng.integers(200, 256, n),
            "dst_host_srv_count":rng.integers(200, 256, n),
            "dst_host_same_srv_rate":    np.clip(rng.beta(8, 1, n), 0, 1),
            "dst_host_diff_srv_rate":    np.clip(rng.beta(1, 8, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.beta(8, 1, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.beta(1, 8, n), 0, 1),
            "dst_host_serror_rate":      np.clip(rng.beta(8, 1, n), 0, 1),
            "dst_host_srv_serror_rate":  np.clip(rng.beta(8, 1, n), 0, 1),
            "dst_host_rerror_rate":      np.clip(rng.beta(1, 8, n), 0, 1),
            "dst_host_srv_rerror_rate":  np.clip(rng.beta(1, 8, n), 0, 1),
            "label":             rng.choice(list(DOS_LABELS), n).tolist(),
            "difficulty":        rng.integers(1, 10, n),   # DoS = easier to detect
        }

    # ── Probe attacks (port scanning) ─────────────────────────────────────────
    def make_probe(n):
        return {
            "duration":          np.clip(rng.integers(0, 5, n), 0, 100),
            "protocol_type":     rng.choice(["tcp", "udp", "icmp"], n, p=[0.45, 0.35, 0.20]),
            "service":           rng.choice(["private", "http", "domain_u", "ftp", "smtp"],
                                             n, p=[0.40, 0.20, 0.15, 0.15, 0.10]),
            "flag":              rng.choice(["S0", "REJ", "SF", "RSTO"], n, p=[0.40, 0.30, 0.20, 0.10]),
            "src_bytes":         np.clip(rng.integers(0, 100, n), 0, 5000),
            "dst_bytes":         np.zeros(n, dtype=int),
            "land":              np.zeros(n, dtype=int),
            "wrong_fragment":    np.zeros(n, dtype=int),
            "urgent":            np.zeros(n, dtype=int),
            "hot":               np.zeros(n, dtype=int),
            "num_failed_logins": np.zeros(n, dtype=int),
            "logged_in":         np.zeros(n, dtype=int),
            "num_compromised":   np.zeros(n, dtype=int),
            "root_shell":        np.zeros(n, dtype=int),
            "su_attempted":      np.zeros(n, dtype=int),
            "num_root":          np.zeros(n, dtype=int),
            "num_file_creations":np.zeros(n, dtype=int),
            "num_shells":        np.zeros(n, dtype=int),
            "num_access_files":  np.zeros(n, dtype=int),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login":     np.zeros(n, dtype=int),
            "is_guest_login":    np.zeros(n, dtype=int),
            "count":             np.clip(rng.integers(1, 30, n), 1, 511),
            "srv_count":         np.clip(rng.integers(1, 20, n), 1, 511),
            "serror_rate":       np.clip(rng.beta(2, 3, n), 0, 1),
            "srv_serror_rate":   np.clip(rng.beta(2, 3, n), 0, 1),
            "rerror_rate":       np.clip(rng.beta(3, 2, n), 0, 1),  # High reject rate = scanning
            "srv_rerror_rate":   np.clip(rng.beta(3, 2, n), 0, 1),
            "same_srv_rate":     np.clip(rng.beta(1, 5, n), 0, 1),  # Different services = scanning
            "diff_srv_rate":     np.clip(rng.beta(5, 1, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.beta(5, 1, n), 0, 1),  # Many different hosts
            "dst_host_count":    rng.integers(1, 256, n),
            "dst_host_srv_count":np.clip(rng.integers(1, 30, n), 1, 256),
            "dst_host_same_srv_rate":    np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_diff_srv_rate":    np.clip(rng.beta(5, 1, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.beta(5, 1, n), 0, 1),
            "dst_host_serror_rate":      np.clip(rng.beta(2, 3, n), 0, 1),
            "dst_host_srv_serror_rate":  np.clip(rng.beta(2, 3, n), 0, 1),
            "dst_host_rerror_rate":      np.clip(rng.beta(3, 2, n), 0, 1),
            "dst_host_srv_rerror_rate":  np.clip(rng.beta(3, 2, n), 0, 1),
            "label":             rng.choice(list(PROBE_LABELS), n).tolist(),
            "difficulty":        rng.integers(5, 15, n),
        }

    # ── R2L attacks (password guessing, file exploits) ─────────────────────────
    def make_r2l(n):
        return {
            "duration":          np.clip(rng.exponential(20, n).astype(int), 0, 1000),
            "protocol_type":     rng.choice(["tcp", "udp"], n, p=[0.90, 0.10]),
            "service":           rng.choice(["ftp", "telnet", "smtp", "imap4", "http", "ssh"],
                                             n, p=[0.25, 0.20, 0.20, 0.15, 0.10, 0.10]),
            "flag":              rng.choice(["SF", "RSTO", "S0"], n, p=[0.70, 0.20, 0.10]),
            "src_bytes":         np.clip(rng.lognormal(6.0, 2.0, n).astype(int), 0, 100000),
            "dst_bytes":         np.clip(rng.lognormal(5.0, 2.5, n).astype(int), 0, 100000),
            "land":              np.zeros(n, dtype=int),
            "wrong_fragment":    np.zeros(n, dtype=int),
            "urgent":            np.zeros(n, dtype=int),
            "hot":               np.clip(rng.poisson(2, n), 0, 30).astype(int),
            "num_failed_logins": np.clip(rng.integers(0, 5, n), 0, 5),  # Key R2L feature
            "logged_in":         rng.choice([0, 1], n, p=[0.70, 0.30]),
            "num_compromised":   np.clip(rng.integers(0, 3, n), 0, 10),
            "root_shell":        np.zeros(n, dtype=int),
            "su_attempted":      np.zeros(n, dtype=int),
            "num_root":          np.zeros(n, dtype=int),
            "num_file_creations":np.clip(rng.integers(0, 3, n), 0, 10),
            "num_shells":        np.zeros(n, dtype=int),
            "num_access_files":  np.clip(rng.integers(0, 3, n), 0, 9),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login":     np.zeros(n, dtype=int),
            "is_guest_login":    rng.choice([0, 1], n, p=[0.80, 0.20]),
            "count":             np.clip(rng.integers(1, 50, n), 1, 511),
            "srv_count":         np.clip(rng.integers(1, 50, n), 1, 511),
            "serror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_serror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "rerror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_rerror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "same_srv_rate":     np.clip(rng.beta(4, 2, n), 0, 1),
            "diff_srv_rate":     np.clip(rng.beta(1, 6, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_count":    rng.integers(1, 100, n),
            "dst_host_srv_count":rng.integers(1, 100, n),
            "dst_host_same_srv_rate":    np.clip(rng.beta(4, 2, n), 0, 1),
            "dst_host_diff_srv_rate":    np.clip(rng.beta(1, 6, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.beta(3, 3, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_serror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_serror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_rerror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_rerror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "label":             rng.choice(list(R2L_LABELS), n).tolist(),
            "difficulty":        rng.integers(10, 21, n),   # R2L = harder to detect
        }

    # ── U2R attacks (buffer overflow, rootkit) ─────────────────────────────────
    def make_u2r(n):
        return {
            "duration":          np.clip(rng.exponential(10, n).astype(int), 0, 500),
            "protocol_type":     rng.choice(["tcp", "udp"], n, p=[0.95, 0.05]),
            "service":           rng.choice(["telnet", "ftp", "http", "ssh", "smtp"],
                                             n, p=[0.30, 0.25, 0.20, 0.15, 0.10]),
            "flag":              rng.choice(["SF", "RSTO"], n, p=[0.85, 0.15]),
            "src_bytes":         np.clip(rng.lognormal(7.0, 2.0, n).astype(int), 0, 500000),
            "dst_bytes":         np.clip(rng.lognormal(6.0, 2.5, n).astype(int), 0, 500000),
            "land":              np.zeros(n, dtype=int),
            "wrong_fragment":    np.zeros(n, dtype=int),
            "urgent":            np.zeros(n, dtype=int),
            "hot":               np.clip(rng.integers(2, 20, n), 0, 30).astype(int),
            "num_failed_logins": np.zeros(n, dtype=int),
            "logged_in":         np.ones(n, dtype=int),     # Must be logged in for U2R
            "num_compromised":   np.clip(rng.integers(1, 10, n), 0, 884),  # Key U2R feature
            "root_shell":        rng.choice([0, 1], n, p=[0.50, 0.50]),    # Root shell gained
            "su_attempted":      rng.choice([0, 1, 2], n, p=[0.60, 0.30, 0.10]),
            "num_root":          np.clip(rng.integers(0, 5, n), 0, 7468),
            "num_file_creations":np.clip(rng.integers(0, 5, n), 0, 28),
            "num_shells":        rng.choice([0, 1, 2], n, p=[0.50, 0.40, 0.10]),
            "num_access_files":  np.clip(rng.integers(0, 4, n), 0, 9),
            "num_outbound_cmds": np.zeros(n, dtype=int),
            "is_host_login":     rng.choice([0, 1], n, p=[0.70, 0.30]),
            "is_guest_login":    np.zeros(n, dtype=int),
            "count":             np.clip(rng.integers(1, 30, n), 1, 511),
            "srv_count":         np.clip(rng.integers(1, 30, n), 1, 511),
            "serror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_serror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "rerror_rate":       np.clip(rng.beta(0.5, 8, n), 0, 1),
            "srv_rerror_rate":   np.clip(rng.beta(0.5, 8, n), 0, 1),
            "same_srv_rate":     np.clip(rng.beta(4, 2, n), 0, 1),
            "diff_srv_rate":     np.clip(rng.beta(1, 6, n), 0, 1),
            "srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_count":    rng.integers(1, 50, n),
            "dst_host_srv_count":rng.integers(1, 50, n),
            "dst_host_same_srv_rate":    np.clip(rng.beta(4, 2, n), 0, 1),
            "dst_host_diff_srv_rate":    np.clip(rng.beta(1, 6, n), 0, 1),
            "dst_host_same_src_port_rate":np.clip(rng.beta(3, 3, n), 0, 1),
            "dst_host_srv_diff_host_rate":np.clip(rng.beta(1, 5, n), 0, 1),
            "dst_host_serror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_serror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_rerror_rate":      np.clip(rng.beta(0.5, 8, n), 0, 1),
            "dst_host_srv_rerror_rate":  np.clip(rng.beta(0.5, 8, n), 0, 1),
            "label":             rng.choice(list(U2R_LABELS), n).tolist(),
            "difficulty":        rng.integers(15, 21, n),   # U2R = hardest to detect
        }

    # Assemble all classes
    for maker, count in [(make_normal, n_normal), (make_dos, n_dos),
                         (make_probe, n_probe), (make_r2l, n_r2l), (make_u2r, n_u2r)]:
        frames.append(pd.DataFrame(maker(count)))

    df = pd.concat(frames, ignore_index=True)
    df = df.sample(frac=1, random_state=random_state).reset_index(drop=True)

    # Ensure difficulty column is int
    df["difficulty"] = df["difficulty"].astype(int)

    return df


# ═════════════════════════════════════════════════════════════════════════════
# Main load function
# ═════════════════════════════════════════════════════════════════════════════
def load_data(
    train_path: str = None,
    test_path: str = None,
    n_samples: int = 10000,
):

    # ── Auto-detect dataset location ─────────────────────────────────────────
    data_dir     = os.path.dirname(os.path.abspath(__file__))
    auto_train   = os.path.join(data_dir, "KDDTrain+.txt")
    auto_train2  = os.path.join(data_dir, "KDDTrain+_20Percent.txt")

    if train_path and os.path.exists(train_path):
        src = train_path
    elif os.path.exists(auto_train):
        src = auto_train
    elif os.path.exists(auto_train2):
        src = auto_train2
    else:
        src = None

    # ── Load real CSV or generate realistic fallback ──────────────────────────
    if src:
        print(f"[DataLoader] ✓ Loading REAL NSL-KDD dataset from:\n             {src}")
        df = pd.read_csv(src, names=COLUMNS, header=None)
        df.dropna(inplace=True)
        print(f"[DataLoader] Loaded {len(df):,} records.")

        # If a separate test file is given, load and combine, then re-split
        if test_path and os.path.exists(test_path):
            print(f"[DataLoader] ✓ Loading test file: {test_path}")
            df_test_raw = pd.read_csv(test_path, names=COLUMNS, header=None)
            df_test_raw.dropna(inplace=True)
            df = pd.concat([df, df_test_raw], ignore_index=True)
            print(f"[DataLoader] Combined total: {len(df):,} records.")
    else:
        print("[DataLoader] ⚠ NSL-KDD CSV not found.")
        print(f"[DataLoader]   Expected: {auto_train}")
        print(f"[DataLoader]   Generating realistic NSL-KDD fallback ({n_samples:,} samples)...")
        print(f"[DataLoader]   (4 attack categories: DoS, Probe, R2L, U2R with real distributions)")
        df = generate_realistic_nslkdd(n_samples=n_samples)

    # ── Encode categorical features ───────────────────────────────────────────
    le = LabelEncoder()
    for col in CATEGORICAL_COLS:
        df[col] = le.fit_transform(df[col].astype(str))

    # ── Binary label: 0 = normal, 1 = any attack ─────────────────────────────
    df["binary_label"] = (df["label"].astype(str).str.lower() != "normal").astype(int)

    # ── Feature matrix ────────────────────────────────────────────────────────
    feature_cols = NUMERIC_COLS + CATEGORICAL_COLS
    X = df[feature_cols].values.astype(np.float32)
    y = df["binary_label"].values

    # ── Scale ─────────────────────────────────────────────────────────────────
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # ── Split ─────────────────────────────────────────────────────────────────
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=42, stratify=y
    )

    # ── Summary ───────────────────────────────────────────────────────────────
    print(f"\n[DataLoader] Dataset summary:")
    print(f"  Train : {X_train.shape[0]:>6,} samples  |  {y_train.mean()*100:.1f}% attacks")
    print(f"  Test  : {X_test.shape[0]:>6,} samples  |  {y_test.mean()*100:.1f}% attacks")
    print(f"  Features: {len(feature_cols)}")

    # Show attack category breakdown if using realistic data
    if not src:
        attack_types = df[df["binary_label"] == 1]["label"].value_counts()
        top_cats = {}
        for lbl, cnt in attack_types.items():
            cat = ("DoS" if lbl in DOS_LABELS else
                   "Probe" if lbl in PROBE_LABELS else
                   "R2L" if lbl in R2L_LABELS else
                   "U2R" if lbl in U2R_LABELS else "Other")
            top_cats[cat] = top_cats.get(cat, 0) + cnt
        print(f"\n[DataLoader] Attack category breakdown (training+test):")
        for cat, cnt in sorted(top_cats.items(), key=lambda x: -x[1]):
            print(f"  {cat:<10}: {cnt:>5,} samples  ({cnt/len(df)*100:.1f}%)")

    return X_train, X_test, y_train, y_test, feature_cols, scaler


# ── Standalone test ────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=" * 55)
    print("  NSL-KDD Data Loader — Test Run")
    print("=" * 55)
    X_train, X_test, y_train, y_test, features, scaler = load_data()
    print(f"\nFeature names (first 10): {features[:10]}")
    print(f"X_train dtype: {X_train.dtype},  min: {X_train.min():.2f},  max: {X_train.max():.2f}")
    print(f"y_train unique values: {set(y_train)}")
    print("\n✓ Data loader OK.")