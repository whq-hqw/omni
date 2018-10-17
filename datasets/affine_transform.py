import tensorflow as tf

class AffineTransform():
    """
    Generate 2D affine transformation matrix
    https://www.mathworks.com/help/images/matrix-representation-of-geometric-transformations.html
    """
    def __init__(self, translation, scale,  shear, rotation, project,
                 custom=None, mean=0, stddev=0.1, order="random"):
        self.order = order
        self.matrices = {
            "translation": translation(translation, mean, stddev),
            "scale": scale(scale, mean, stddev),
            "shear": shear(shear, mean, stddev),
            "rotation": rotation(rotation, mean, stddev),
            "project": project(project, mean, stddev)
        }
        if custom:
            assert type(custom) is list and len(custom) is 8, \
                "custom transformation matrix setting error"
            self.matrices.update({"custom": tf.constant(custom, dtype=tf.float32)})

    @staticmethod
    def translation(trans, mean, stddev):
        """Generate the translation matrix with a little noise."""
        if trans is None:
            return None
        assert len(trans) is 2, "translation setting error"
        base_mat = tf.constant([1, 0, 0, 0, 1, 0, trans[0], trans[1]], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0,0,0,0,0,0,1,1], dtype=tf.float32)
        return tf.multiply(base_mat,tf.multiply(rand, mask))

    @staticmethod
    def scale(scale, mean, stddev):
        """Generate the scale matrix with a little noise."""
        if scale is None:
            return None
        assert len(scale) is 2, "scale setting error"
        base_mat = tf.constant([scale[0], 0, 0, 0, scale[1], 0, 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    @staticmethod
    def shear(shear, mean, stddev):
        """Generate the shear matrix with a little noise."""
        if shear is None:
            return None
        assert len(shear) is 2, "shear setting error"
        base_mat = tf.constant([1, shear[0], 0, shear[1], 1, 0, 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0, 1, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    @staticmethod
    def rotation(rot, mean, stddev):
        """Generate the rotation matrix with a little noise."""
        if rot is None:
            return None
        assert len(rot) is 1, "rotation setting error"
        rand = tf.random_normal(shape=[1], mean=mean, stddev=stddev)
        rot = tf.multiply(tf.constant(rot, dtype=tf.float32), rand)
        return tf.constant([tf.cos(rot[0]), tf.sin(rot[0]), 0, -tf.sin(rot[0]), tf.cos(rot[0]), 0, 0, 0],
                           dtype=tf.float32)

    @staticmethod
    def project(project, mean, stddev):
        """Generate the projective matrix with a little noise."""
        if project is None:
            return None
        assert len(project) is 2, "project setting error"
        base_mat = tf.constant([1, 0, project[0], 0, 1, project[1], 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    def to_transform_matrix(self):
        if self.order is "random":
            values = tf.random_shuffle(list(self.matrices.values()))
        elif type(self.order) is list:
            # TODO: Think about when length of self.order is not equal to self.matrices
            values = [self.matrices[key] for key in self.matrices]
        else:
            values = [self.matrices[key] for key in self.matrices]
        return tf.concat(values)