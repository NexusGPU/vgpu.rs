use ctor::ctor;
use std::env;

fn set_env_var(env_var: &str, tf_var: &str) {
    if let Ok(tf_value) = env::var(tf_var) {
        let is_prepend = tf_var.starts_with("TF_PREPEND_");
        if let Ok(current_value) = env::var(env_var) {
            if !current_value.split(':').any(|p| p == tf_value) {
                if is_prepend {
                    env::set_var(env_var, format!("{}:{}", tf_value, current_value));
                } else {
                    env::set_var(env_var, format!("{}:{}", current_value, tf_value));
                }
            }
        } else {
            env::set_var(env_var, tf_value);
        }
    }
}

#[ctor]
unsafe fn entry_point() {
    set_env_var("PATH", "TF_PATH");
    set_env_var("LD_PRELOAD", "TF_LD_PRELOAD");
    set_env_var("LD_LIBRARY_PATH", "TF_LD_LIBRARY_PATH");

    set_env_var("PATH", "TF_PREPEND_PATH");
    set_env_var("LD_PRELOAD", "TF_PREPEND_LD_PRELOAD");
    set_env_var("LD_LIBRARY_PATH", "TF_PREPEND_LD_LIBRARY_PATH");
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::env;

    #[test]
    fn test_set_env_var_if_not_exists_adds_new_var() {
        env::set_var("TF_TEST_PATH_1", "/new/path");
        set_env_var("TEST_PATH_1", "TF_TEST_PATH_1");
        assert_eq!(env::var("TEST_PATH_1").unwrap(), "/new/path");
        env::remove_var("TEST_PATH_1");
        env::remove_var("TF_TEST_PATH_1");
    }

    #[test]
    fn test_set_env_var_if_not_exists_does_not_duplicate() {
        env::set_var("TEST_PATH_2", "/existing/path");
        env::set_var("TF_TEST_PATH_2", "/existing/path");
        set_env_var("TEST_PATH_2", "TF_TEST_PATH_2");
        assert_eq!(env::var("TEST_PATH_2").unwrap(), "/existing/path");
        env::remove_var("TEST_PATH_2");
        env::remove_var("TF_TEST_PATH_2");
    }

    #[test]
    fn test_set_env_var_if_not_exists_appends_var() {
        env::set_var("TEST_PATH_3", "/existing/path");
        env::set_var("TF_TEST_PATH_3", "/new/path");
        set_env_var("TEST_PATH_3", "TF_TEST_PATH_3");
        assert_eq!(env::var("TEST_PATH_3").unwrap(), "/existing/path:/new/path");
        env::remove_var("TEST_PATH_3");
        env::remove_var("TF_TEST_PATH_3");
    }

    #[test]
    fn test_set_env_var_if_not_exists_prepends_var() {
        env::set_var("TEST_PATH_4", "/existing/path");
        env::set_var("TF_PREPEND_TEST_PATH_4", "/new/path");
        set_env_var("TEST_PATH_4", "TF_PREPEND_TEST_PATH_4");
        assert_eq!(env::var("TEST_PATH_4").unwrap(), "/new/path:/existing/path");
        env::remove_var("TEST_PATH_4");
        env::remove_var("TF_PREPEND_TEST_PATH_4");
    }

    #[test]
    fn test_set_env_var_if_not_exists_prepends_var_no_duplicate() {
        env::set_var("TEST_PATH_5", "/existing/path");
        env::set_var("TF_PREPEND_TEST_PATH_5", "/existing/path");
        set_env_var("TEST_PATH_5", "TF_PREPEND_TEST_PATH_5");
        assert_eq!(env::var("TEST_PATH_5").unwrap(), "/existing/path");
        env::remove_var("TEST_PATH_5");
        env::remove_var("TF_PREPEND_TEST_PATH_5");
    }
}
