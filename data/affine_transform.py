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
            "translation": self._translation(translation, mean, stddev),
            "scale": self._scale(scale, mean, stddev),
            "shear": self._shear(shear, mean, stddev),
            "rotation": self._rotation(rotation, mean, stddev),
            "project": self._project(project, mean, stddev)
        }
        if custom:
            assert type(custom) is list and len(custom) is 8, \
                "custom transformation matrix setting error"
            self.matrices.update({"custom": tf.constant(custom, dtype=tf.float32)})

    @staticmethod
    def _translation(trans, mean, stddev):
        """Generate the translation matrix with a little noise."""
        if trans is None:
            return None
        assert len(trans) is 2, "translation setting error"
        base_mat = tf.constant([1, 0, 0, 0, 1, 0, trans[0], trans[1]], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0,0,0,0,0,0,1,1], dtype=tf.float32)
        return tf.multiply(base_mat,tf.multiply(rand, mask))

    @staticmethod
    def _scale(scale, mean, stddev):
        """Generate the scale matrix with a little noise."""
        if scale is None:
            return None
        assert len(scale) is 2, "scale setting error"
        base_mat = tf.constant([scale[0], 0, 0, 0, scale[1], 0, 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([1, 0, 0, 0, 1, 0, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    @staticmethod
    def _shear(shear, mean, stddev):
        """Generate the shear matrix with a little noise."""
        if shear is None:
            return None
        assert len(shear) is 2, "shear setting error"
        base_mat = tf.constant([1, shear[0], 0, shear[1], 1, 0, 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0, 1, 0, 1, 0, 0, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    @staticmethod
    def _rotation(rot, mean, stddev):
        """Generate the rotation matrix with a little noise."""
        if rot is None:
            return None
        assert len(rot) is 1, "rotation setting error"
        rand = tf.random_normal(shape=[1], mean=mean, stddev=stddev)
        rot = tf.multiply(tf.constant(rot, dtype=tf.float32), rand)
        return tf.stack([tf.cos(rot[0]), tf.sin(rot[0]), 0, -tf.sin(rot[0]), tf.cos(rot[0]), 0, 0, 0])

    @staticmethod
    def _project(project, mean, stddev):
        """Generate the projective matrix with a little noise."""
        if project is None:
            return None
        assert len(project) is 2, "project setting error"
        base_mat = tf.constant([1, 0, project[0], 0, 1, project[1], 0, 0], dtype=tf.float32)
        rand = tf.random_normal(shape=[8], mean=mean, stddev=stddev)
        mask = tf.constant([0, 0, 1, 0, 0, 1, 0, 0], dtype=tf.float32)
        return tf.multiply(base_mat, tf.multiply(rand, mask))

    def to_transform_matrix(self):
        """
        :return: a random ordered transformation matrix
        """
        values = tf.random_shuffle(list(self.matrices.values()))
        return tf.concat(values, axis=1)
    
if __name__ == "__main__":
    """
        Test Code
    """
    affine = AffineTransform(translation=(10, 10), scale=(1, 1),  shear=(2, 2), rotation=[5.0],
                             project=(2, 2), custom=None, mean=0.0, stddev=0.1, order="random")
    affine_mat = affine.to_transform_matrix()
    sess = tf.Session()
    
    mat = sess.run(affine_mat)
    print(mat)
    